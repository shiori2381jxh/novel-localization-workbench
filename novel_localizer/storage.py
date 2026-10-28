from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from ctypes import wintypes
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Iterable

from .domain import (
    DEFAULT_CATEGORIES,
    TRANSLATION_MODES,
    forced_rename_validation_error,
    ImportResult,
    ValidationError,
)


SCHEMA_VERSION = 1
PROJECT_MANIFEST = "project.json"
ATOMIC_REPLACE_ATTEMPTS = 5
ATOMIC_REPLACE_BASE_DELAY_SECONDS = 0.01


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _replace_with_retry(source, destination) -> None:
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 >= ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(ATOMIC_REPLACE_BASE_DELAY_SECONDS * (2**attempt))


def _project_transaction(method):
    @wraps(method)
    def locked(self, project_id, *args, **kwargs):
        with self.project_lock(project_id):
            return method(self, project_id, *args, **kwargs)

    return locked


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        _replace_with_retry(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        _replace_with_retry(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _safe_filename(value: str, fallback: str = "未命名") -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value.strip())
    value = value.rstrip(". ")
    return value[:80] or fallback


def _normalized_categories(categories: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    seen_filenames: set[str] = set()
    for raw in categories:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        if len(value) > 30:
            raise ValidationError(f"替换类别“{value}”过长，请控制在 30 个字以内。")
        filename_key = _safe_filename(value, "其他").casefold()
        if filename_key in seen_filenames:
            raise ValidationError(
                f"替换类别“{value}”与另一类别会使用相同文件名，请换一个名称。"
            )
        seen.add(value)
        seen_filenames.add(filename_key)
        result.append(value)
    return result or list(DEFAULT_CATEGORIES)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5", "shift_jis"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class ProjectStore:
    """管理项目、章节和项目级替换清单。"""

    def __init__(self, data_root: Path | str):
        self.data_root = Path(data_root)
        self.projects_root = self.data_root / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self._project_locks: dict[str, threading.RLock] = {}
        self._project_locks_guard = threading.Lock()

    def project_lock(self, project_id: str) -> threading.RLock:
        self.project_dir(project_id)
        with self._project_locks_guard:
            return self._project_locks.setdefault(project_id, threading.RLock())

    def list_projects(self) -> list[dict]:
        projects: list[dict] = []
        for manifest_path in self.projects_root.glob(f"*/{PROJECT_MANIFEST}"):
            try:
                manifest = _read_json(manifest_path)
                if isinstance(manifest, dict) and manifest.get("id"):
                    projects.append(manifest)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(
            projects,
            key=lambda item: (item.get("updated_at", ""), item.get("name", "")),
            reverse=True,
        )

    def project_dir(self, project_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", project_id):
            raise ValidationError("项目编号不合法。")
        path = (self.projects_root / project_id).resolve()
        if self.projects_root.resolve() not in path.parents:
            raise ValidationError("项目路径不合法。")
        return path

    @_project_transaction
    def refresh_completed_output(self, project_id: str) -> Path:
        """Refresh the user-facing folder that contains only completed chapter text."""
        project_dir = self.project_dir(project_id)
        output_dir = (project_dir / "output").resolve()
        completed_dir = project_dir / "已完成译文"
        completed_dir.mkdir(parents=True, exist_ok=True)
        index_path = completed_dir / ".novel_localizer_completed_output.json"
        previous_index = _read_json(index_path, {})
        if not isinstance(previous_index, dict):
            previous_index = {}
        previous_names = {
            str(name)
            for name in previous_index.get("files", [])
            if isinstance(name, str)
        }

        current_names: set[str] = set()
        manifest = self.load_project(project_id)
        for chapter in manifest.get("chapters", []):
            translations = chapter.get("translations", [])
            if chapter.get("status") != "已翻译" or not translations:
                continue
            output_name = str(translations[-1].get("output_name", ""))
            source_path = (output_dir / output_name).resolve()
            if (
                not output_name
                or source_path.parent != output_dir
                or not source_path.is_file()
            ):
                continue
            shutil.copy2(source_path, completed_dir / source_path.name)
            current_names.add(source_path.name)

        completed_dir_resolved = completed_dir.resolve()
        for stale_name in previous_names - current_names:
            stale_path = (completed_dir / stale_name).resolve()
            if stale_path.parent == completed_dir_resolved and stale_path.is_file():
                stale_path.unlink()
        _atomic_write_json(index_path, {"files": sorted(current_names)})
        return completed_dir

    def load_project(self, project_id: str) -> dict:
        manifest_path = self.project_dir(project_id) / PROJECT_MANIFEST
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValidationError("项目不存在或项目资料已损坏。")
        return manifest

    def load_translation_checkpoint(
        self, project_id: str, chapter_id: str
    ) -> dict | None:
        checkpoint = _read_json(self._translation_checkpoint_path(project_id, chapter_id))
        return checkpoint if isinstance(checkpoint, dict) else None

    @_project_transaction
    def save_translation_checkpoint(
        self, project_id: str, chapter_id: str, checkpoint: dict
    ) -> None:
        if not isinstance(checkpoint, dict):
            raise ValidationError("翻译检查点格式不正确。")
        _atomic_write_json(self._translation_checkpoint_path(project_id, chapter_id), checkpoint)

    @_project_transaction
    def clear_translation_checkpoint(self, project_id: str, chapter_id: str) -> None:
        path = self._translation_checkpoint_path(project_id, chapter_id)
        if path.exists():
            path.unlink()

    def _translation_checkpoint_path(self, project_id: str, chapter_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", chapter_id):
            raise ValidationError("章节编号不合法。")
        return self.project_dir(project_id) / "checkpoints" / f"{chapter_id}.translation.json"

    @_project_transaction
    def update_processing_settings(self, project_id: str, settings: dict) -> dict:
        """Persist the translation controls selected for one project."""
        if not isinstance(settings, dict):
            raise ValidationError("项目处理配置格式不正确。")
        manifest = self.load_project(project_id)
        current = dict(manifest.get("processing_settings", {}))
        current.update({
            key: value for key, value in settings.items()
            if key in {"mode", "max_output_tokens", "profile_id", "extra_instructions", "save_extra_instructions"}
        })
        manifest["processing_settings"] = current
        manifest["updated_at"] = utc_now()
        _atomic_write_json(self.project_dir(project_id) / PROJECT_MANIFEST, manifest)
        return manifest

    def proper_noun_localization_enabled(self, project_id: str) -> bool:
        """Return the project's forced proper-name localization policy."""
        manifest = self.load_project(project_id)
        value = manifest.get("proper_noun_localization", True)
        return value if isinstance(value, bool) else True

    @_project_transaction
    def update_project_settings(
        self, project_id: str, *, proper_noun_localization: bool
    ) -> dict:
        if not isinstance(proper_noun_localization, bool):
            raise ValidationError("专名本地化/改名设置格式不正确。")
        manifest = self.load_project(project_id)
        manifest["proper_noun_localization"] = proper_noun_localization
        manifest["updated_at"] = utc_now()
        _atomic_write_json(self.project_dir(project_id) / PROJECT_MANIFEST, manifest)
        return manifest

    def create_project(
        self,
        name: str,
        source_language: str,
        target_language: str,
        categories: Iterable[str],
        skill_paths: Iterable[Path | str] = (),
        processing_mode: str | None = None,
        proper_noun_localization: bool = True,
    ) -> dict:
        clean_name = name.strip()
        if not clean_name:
            raise ValidationError("请输入项目名称。")
        if len(clean_name) > 80:
            raise ValidationError("项目名称请控制在 80 个字以内。")
        if not source_language.strip() or not target_language.strip():
            raise ValidationError("请选择原文语言和目标语言。")
        if processing_mode is not None and processing_mode not in TRANSLATION_MODES:
            raise ValidationError("请选择有效的处理模式。")
        if not isinstance(proper_noun_localization, bool):
            raise ValidationError("专名本地化/改名设置格式不正确。")

        category_names = _normalized_categories(categories)
        project_id = (
            datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        project_dir = self.project_dir(project_id)
        for folder in ("source", "output", "skills", "replacements"):
            (project_dir / folder).mkdir(parents=True, exist_ok=False)

        now = utc_now()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "id": project_id,
            "name": clean_name,
            "source_language": source_language.strip(),
            "target_language": target_language.strip(),
            "categories": category_names,
            "chapters": [],
            "skills": [],
            "proper_noun_localization": proper_noun_localization,
            "created_at": now,
            "updated_at": now,
        }
        if processing_mode is not None:
            manifest["processing_mode"] = processing_mode

        try:
            for category in category_names:
                self._create_registry(project_dir, category)
            manifest["skills"] = self._copy_skills(project_dir, skill_paths)
            _atomic_write_json(project_dir / PROJECT_MANIFEST, manifest)
        except Exception:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise
        return manifest

    @_project_transaction
    def set_processing_mode(self, project_id: str, mode: str) -> dict:
        if mode not in TRANSLATION_MODES:
            raise ValidationError("请选择有效的处理模式。")
        manifest = self.load_project(project_id)
        existing = manifest.get("processing_mode")
        if existing in TRANSLATION_MODES:
            if existing != mode:
                raise ValidationError("项目处理模式已锁定，不能修改。")
            return manifest
        manifest["processing_mode"] = mode
        manifest["updated_at"] = utc_now()
        _atomic_write_json(self.project_dir(project_id) / PROJECT_MANIFEST, manifest)
        return manifest

    def pending_skill_analysis(self, project_id: str) -> list[dict]:
        """Return Skills whose cached analysis is absent or from different content."""
        project = self.load_project(project_id)
        return [
            dict(skill) for skill in project.get("skills", [])
            if not isinstance(skill.get("analysis"), dict)
            or skill["analysis"].get("source_hash") != skill.get("sha256")
        ]

    @_project_transaction
    def save_skill_analysis(
        self, project_id: str, skill_id: str, source_hash: str, rules: dict
    ) -> dict:
        if not isinstance(rules, dict):
            raise ValidationError("Skill 解析结果格式不正确。")
        manifest = self.load_project(project_id)
        skill = next(
            (item for item in manifest.get("skills", []) if item.get("id") == skill_id),
            None,
        )
        if not skill:
            raise ValidationError("找不到所选 Skill。")
        if skill.get("sha256") != source_hash:
            raise ValidationError("Skill 内容已变化，请重新解析。")
        skill["analysis"] = {
            "source_hash": source_hash,
            "rules": rules,
            "analyzed_at": utc_now(),
        }
        manifest["updated_at"] = utc_now()
        _atomic_write_json(self.project_dir(project_id) / PROJECT_MANIFEST, manifest)
        return dict(skill)

    @_project_transaction
    def import_chapters(
        self, project_id: str, chapter_paths: Iterable[Path | str]
    ) -> ImportResult:
        project_dir = self.project_dir(project_id)
        manifest = self.load_project(project_id)
        known_hashes = {
            chapter.get("sha256")
            for chapter in manifest.get("chapters", [])
            if chapter.get("sha256")
        }
        imported: list[dict] = []
        skipped: list[dict] = []

        for raw_path in chapter_paths:
            source_path = Path(raw_path)
            if not source_path.is_file():
                raise ValidationError(f"找不到 TXT 文件：{source_path}")
            if source_path.suffix.lower() != ".txt":
                raise ValidationError(f"只支持导入 TXT 文件：{source_path.name}")

            data = source_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest in known_hashes:
                skipped.append({"source_name": source_path.name, "reason": "内容已导入"})
                continue

            destination = self._available_destination(
                project_dir / "source", source_path.name
            )
            shutil.copy2(source_path, destination)
            text = _decode_text(data)
            record = {
                "id": uuid.uuid4().hex,
                "source_name": source_path.name,
                "stored_name": destination.name,
                "sha256": digest,
                "bytes": len(data),
                "characters": len(text),
                "status": "待处理",
                "name_mappings": {},
                "imported_at": utc_now(),
            }
            manifest.setdefault("chapters", []).append(record)
            imported.append(record)
            known_hashes.add(digest)

        if imported:
            manifest["updated_at"] = utc_now()
            _atomic_write_json(project_dir / PROJECT_MANIFEST, manifest)
        return ImportResult(tuple(imported), tuple(skipped))

    @_project_transaction
    def import_chapter_text(self, project_id: str, text: str) -> ImportResult:
        content = text.strip()
        if not content:
            return ImportResult((), ())
        project_dir = self.project_dir(project_id)
        manifest = self.load_project(project_id)
        data = content.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        known_hashes = {
            chapter.get("sha256")
            for chapter in manifest.get("chapters", [])
            if chapter.get("sha256")
        }
        if digest in known_hashes:
            return ImportResult(
                (), ({"source_name": "粘贴章节", "reason": "内容已导入"},)
            )

        number = 1
        while (project_dir / "source" / f"粘贴章节 {number}.txt").exists():
            number += 1
        source_name = f"粘贴章节 {number}.txt"
        _atomic_write_text(project_dir / "source" / source_name, content)
        record = {
            "id": uuid.uuid4().hex,
            "source_name": source_name,
            "stored_name": source_name,
            "sha256": digest,
            "bytes": len(data),
            "characters": len(content),
            "status": "待处理",
            "name_mappings": {},
            "imported_at": utc_now(),
        }
        manifest.setdefault("chapters", []).append(record)
        manifest["updated_at"] = utc_now()
        _atomic_write_json(project_dir / PROJECT_MANIFEST, manifest)
        return ImportResult((record,), ())

    @_project_transaction
    def add_skills(
        self, project_id: str, skill_paths: Iterable[Path | str]
    ) -> list[dict]:
        project_dir = self.project_dir(project_id)
        manifest = self.load_project(project_id)
        existing_hashes = {item.get("sha256") for item in manifest.get("skills", [])}
        added = self._copy_skills(
            project_dir, skill_paths, skip_hashes=existing_hashes
        )
        if added:
            manifest.setdefault("skills", []).extend(added)
            manifest["updated_at"] = utc_now()
            _atomic_write_json(project_dir / PROJECT_MANIFEST, manifest)
        return added

    @_project_transaction
    def add_skill_text(self, project_id: str, text: str) -> list[dict]:
        content = text.strip()
        if not content:
            return []
        project_dir = self.project_dir(project_id)
        manifest = self.load_project(project_id)
        data = content.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        if digest in {
            item.get("sha256") for item in manifest.get("skills", [])
        }:
            return []

        number = 1
        while (project_dir / "skills" / f"粘贴规则 {number}.md").exists():
            number += 1
        source_name = f"粘贴规则 {number}.md"
        _atomic_write_text(project_dir / "skills" / source_name, content)
        record = {
            "id": uuid.uuid4().hex,
            "source_name": source_name,
            "stored_name": source_name,
            "sha256": digest,
            "added_at": utc_now(),
        }
        manifest.setdefault("skills", []).append(record)
        manifest["updated_at"] = utc_now()
        _atomic_write_json(project_dir / PROJECT_MANIFEST, manifest)
        return [record]

    @_project_transaction
    def update_skill(self, project_id: str, skill_id: str, text: str) -> dict:
        content = text.strip()
        if not content:
            raise ValidationError("Skill 内容不能为空。")
        project_dir = self.project_dir(project_id)
        manifest = self.load_project(project_id)
        skills = manifest.get("skills", [])
        record = next((item for item in skills if item.get("id") == skill_id), None)
        if not record:
            raise ValidationError("找不到所选 Skill。")
        data = content.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        if any(
            item.get("id") != skill_id and item.get("sha256") == digest
            for item in skills
        ):
            raise ValidationError("项目中已经存在内容相同的 Skill。")
        path = self._skill_path(project_dir, record)
        if not path.is_file():
            raise ValidationError(f"Skill 文件不存在：{record.get('source_name', '')}")
        _atomic_write_text(path, content)
        record["sha256"] = digest
        record["updated_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        _atomic_write_json(project_dir / PROJECT_MANIFEST, manifest)
        return dict(record)

    @_project_transaction
    def delete_skills(
        self, project_id: str, skill_ids: Iterable[str]
    ) -> list[dict]:
        project_dir = self.project_dir(project_id)
        manifest = self.load_project(project_id)
        selected = {str(skill_id) for skill_id in skill_ids}
        deleted = [
            item for item in manifest.get("skills", []) if item.get("id") in selected
        ]
        if not deleted:
            return []
        for record in deleted:
            path = self._skill_path(project_dir, record)
            if path.is_file():
                path.unlink()
        deleted_ids = {item.get("id") for item in deleted}
        manifest["skills"] = [
            item
            for item in manifest.get("skills", [])
            if item.get("id") not in deleted_ids
        ]
        manifest["updated_at"] = utc_now()
        _atomic_write_json(project_dir / PROJECT_MANIFEST, manifest)
        return [dict(item) for item in deleted]

    @_project_transaction
    def ensure_categories(
        self, project_id: str, categories: Iterable[str]
    ) -> list[str]:
        project_dir = self.project_dir(project_id)
        manifest = self.load_project(project_id)
        old = list(manifest.get("categories", []))
        combined = _normalized_categories([*old, *categories])
        added = [category for category in combined if category not in old]
        for category in added:
            self._create_registry(project_dir, category)
        if added:
            manifest["categories"] = combined
            manifest["updated_at"] = utc_now()
            _atomic_write_json(project_dir / PROJECT_MANIFEST, manifest)
        return added

    def load_registry(self, project_id: str, category: str) -> dict:
        project_dir = self.project_dir(project_id)
        manifest = self.load_project(project_id)
        if category not in manifest.get("categories", []):
            raise ValidationError(f"项目中没有“{category}”替换类别。")
        path = self._registry_path(project_dir, category)
        registry = _read_json(path)
        if not isinstance(registry, dict):
            raise ValidationError(f"“{category}”替换清单已损坏。")
        return registry

    def load_all_replacements(self, project_id: str) -> dict[str, dict]:
        manifest = self.load_project(project_id)
        return {
            category: self.load_registry(project_id, category)
            for category in manifest.get("categories", [])
        }

    def chapter_mappings(self, project_id: str, chapter_id: str) -> dict[str, dict]:
        chapter = self.get_chapter(project_id, chapter_id)
        mappings = chapter.get("name_mappings", {})
        return json.loads(json.dumps(mappings, ensure_ascii=False)) if isinstance(mappings, dict) else {}

    @_project_transaction
    def upsert_chapter_mapping(
        self, project_id: str, chapter_id: str, category: str, source: str,
        target: str, *, notes: str = "",
    ) -> dict:
        manifest = self.load_project(project_id)
        if category not in manifest.get("categories", []):
            raise ValidationError(f"项目中没有“{category}”替换类别。")
        source_value, target_value = source.strip(), target.strip()
        if not source_value or not target_value:
            raise ValidationError("原词和替换词都不能为空。")
        error = forced_rename_validation_error(category, source_value, target_value)
        if error:
            raise ValidationError(error)
        chapter = next((item for item in manifest.get("chapters", []) if item.get("id") == chapter_id), None)
        if not chapter:
            raise ValidationError("找不到所选章节。")
        entries = chapter.setdefault("name_mappings", {}).setdefault(category, {})
        record = {
            "source": source_value, "target": target_value, "notes": notes.strip(),
            "status": "已确认", "updated_at": utc_now(),
        }
        entries[source_value] = record
        manifest["updated_at"] = utc_now()
        _atomic_write_json(self.project_dir(project_id) / PROJECT_MANIFEST, manifest)
        return dict(record)

    def chapter_prompt_mappings(self, project_id: str, chapter_id: str) -> dict[str, list[dict]]:
        return {
            category: [dict(record) for record in entries.values() if record.get("status") == "已确认"]
            for category, entries in self.chapter_mappings(project_id, chapter_id).items()
            if isinstance(entries, dict)
        }

    def chapter_replacements(self, project_id: str, chapter_id: str) -> list[dict]:
        rows: list[dict] = []
        for category, entries in self.chapter_mappings(project_id, chapter_id).items():
            if not isinstance(entries, dict):
                continue
            for source, record in entries.items():
                if not isinstance(record, dict):
                    continue
                rows.append(
                    {
                        "category": category,
                        "source": str(record.get("source", source)),
                        "target": str(record.get("target", "")),
                        "notes": str(record.get("notes", "")),
                        "status": str(record.get("status", "")),
                    }
                )
        return sorted(rows, key=lambda item: (item["category"], item["source"]))

    @_project_transaction
    def delete_chapters(
        self, project_id: str, chapter_ids: Iterable[str]
    ) -> list[dict]:
        ids = {str(item) for item in chapter_ids if str(item)}
        if not ids:
            return []
        project_dir = self.project_dir(project_id)
        manifest = self.load_project(project_id)
        deleted = [
            item for item in manifest.get("chapters", []) if item.get("id") in ids
        ]
        if not deleted:
            return []

        source_dir = (project_dir / "source").resolve()
        output_dir = (project_dir / "output").resolve()
        for chapter in deleted:
            source_path = (source_dir / str(chapter.get("stored_name", ""))).resolve()
            if source_path.parent == source_dir and source_path.is_file():
                source_path.unlink()
            for translation in chapter.get("translations", []):
                for field in ("output_name", "metadata_name"):
                    path = (output_dir / str(translation.get(field, ""))).resolve()
                    if path.parent == output_dir and path.is_file():
                        path.unlink()

        for category, registry in self.load_all_replacements(project_id).items():
            entries = registry.get("entries", {})
            changed = False
            for source in list(entries):
                record = entries[source]
                old_refs = list(record.get("source_chapters", []))
                new_refs = [item for item in old_refs if item not in ids]
                if new_refs != old_refs:
                    changed = True
                    if new_refs:
                        record["source_chapters"] = new_refs
                    else:
                        del entries[source]
            if changed:
                registry["updated_at"] = utc_now()
                _atomic_write_json(self._registry_path(project_dir, category), registry)

        memory_path = project_dir / "memory" / "project_memory.json"
        memory = _read_json(memory_path)
        if isinstance(memory, dict):
            memory["entities"] = [
                item for item in memory.get("entities", [])
                if not (
                    item.get("source_chapters")
                    and not [ref for ref in item.get("source_chapters", []) if ref not in ids]
                )
            ]
            for item in memory.get("entities", []):
                item["source_chapters"] = [
                    ref for ref in item.get("source_chapters", []) if ref not in ids
                ]
            for field in ("chapter_memory", "draft_chapter_memory"):
                if isinstance(memory.get(field), dict):
                    for chapter_id in ids:
                        memory[field].pop(chapter_id, None)
            for field in ("preparation", "draft_preparation"):
                chapters = memory.get(field, {}).get("chapters", {})
                if isinstance(chapters, dict):
                    for chapter_id in ids:
                        chapters.pop(chapter_id, None)
            memory["version"] = int(memory.get("version", 0)) + 1
            memory["updated_at"] = utc_now()
            _atomic_write_json(memory_path, memory)

        manifest["chapters"] = [
            item for item in manifest.get("chapters", []) if item.get("id") not in ids
        ]
        manifest["updated_at"] = utc_now()
        _atomic_write_json(project_dir / PROJECT_MANIFEST, manifest)
        return [dict(item) for item in deleted]

    def get_chapter(self, project_id: str, chapter_id: str) -> dict:
        manifest = self.load_project(project_id)
        chapter = next(
            (
                item
                for item in manifest.get("chapters", [])
                if item.get("id") == chapter_id
            ),
            None,
        )
        if not chapter:
            raise ValidationError("找不到所选章节。")
        return chapter

    def read_chapter_text(self, project_id: str, chapter_id: str) -> str:
        chapter = self.get_chapter(project_id, chapter_id)
        path = self.project_dir(project_id) / "source" / chapter["stored_name"]
        if not path.is_file():
            raise ValidationError(f"章节原文文件不存在：{chapter['source_name']}")
        return _decode_text(path.read_bytes())

    def read_project_skills(self, project_id: str) -> list[dict]:
        manifest = self.load_project(project_id)
        skills_dir = self.project_dir(project_id) / "skills"
        result: list[dict] = []
        for record in manifest.get("skills", []):
            path = self._skill_path(self.project_dir(project_id), record)
            result.append(
                {
                    "id": record.get("id"),
                    "name": record.get("source_name") or path.name,
                    "source_name": record.get("source_name") or path.name,
                    "stored_name": record.get("stored_name") or path.name,
                    "content": _decode_text(path.read_bytes()) if path.is_file() else "",
                    "missing": not path.is_file(),
                }
            )
        return result

    def _skill_path(self, project_dir: Path, record: dict) -> Path:
        skills_dir = (project_dir / "skills").resolve()
        stored_name = str(record.get("stored_name", ""))
        path = (skills_dir / stored_name).resolve()
        if not stored_name or path.parent != skills_dir:
            raise ValidationError("Skill 文件路径无效。")
        return path

    @_project_transaction
    def save_translation(
        self,
        project_id: str,
        chapter_id: str,
        translated_text: str,
        *,
        profile_id: str,
        provider: str,
        model: str,
        usage: dict | None = None,
        replacements_added: int = 0,
        warnings: Iterable[str] = (),
        mode: str = "",
        review_status: str = "完成",
        residual_sources: Iterable[str] = (),
        memory_version: int = 0,
        skill_hash: str = "",
        skill_files: Iterable[str] = (),
        allow_empty: bool = False,
    ) -> dict:
        clean_text = translated_text.strip()
        if not clean_text and not allow_empty:
            raise ValidationError("译文为空，无法保存。")

        project_dir = self.project_dir(project_id)
        manifest = self.load_project(project_id)
        chapter = next(
            (
                item
                for item in manifest.get("chapters", [])
                if item.get("id") == chapter_id
            ),
            None,
        )
        if not chapter:
            raise ValidationError("找不到要保存译文的章节。")

        output_dir = project_dir / "output"
        output_dir_resolved = output_dir.resolve()
        previous_records = list(chapter.get("translations", []))

        def output_file(name: object) -> Path | None:
            path = (output_dir / str(name or "")).resolve()
            return path if path.parent == output_dir_resolved else None

        latest_record = previous_records[-1] if previous_records else {}
        output_path = output_file(latest_record.get("output_name"))
        if output_path is None:
            source_stem = Path(chapter.get("source_name", "章节")).stem
            output_path = self._available_destination(
                output_dir, f"{source_stem}_译文.txt"
            )
        metadata_path = output_file(latest_record.get("metadata_name"))
        if metadata_path is None:
            metadata_path = output_path.with_suffix(".translation.json")

        previous_files = {
            path: path.read_text(encoding="utf-8")
            for record in previous_records
            for path in (
                output_file(record.get("output_name")),
                output_file(record.get("metadata_name")),
            )
            if path is not None and path.is_file()
        }
        try:
            _atomic_write_text(output_path, clean_text + "\n")
            now = utc_now()
            record = {
                "id": uuid.uuid4().hex,
                "output_name": output_path.name,
                "profile_id": profile_id,
                "provider": provider,
                "model": model,
                "usage": usage or {},
                "replacements_added": replacements_added,
                "warnings": [str(item) for item in warnings],
                "mode": mode,
                "review_status": review_status,
                "residual_sources": [str(item) for item in residual_sources],
                "memory_version": int(memory_version),
                "skill_hash": skill_hash,
                "skill_files": [str(item) for item in skill_files],
                "created_at": now,
            }
            _atomic_write_json(
                metadata_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "project_id": project_id,
                    "chapter_id": chapter_id,
                    **record,
                },
            )
            record["metadata_name"] = metadata_path.name
            chapter["translations"] = [record]
            chapter["status"] = "需要复核" if review_status == "需要复核" else "已翻译"
            chapter["last_error"] = ""
            chapter["translated_at"] = now
            manifest["updated_at"] = now
            _atomic_write_json(project_dir / PROJECT_MANIFEST, manifest)
        except Exception:
            for created_path in (metadata_path, output_path):
                if created_path in previous_files:
                    try:
                        _atomic_write_text(created_path, previous_files[created_path])
                    except OSError:
                        pass
                    continue
                try:
                    if created_path.is_file():
                        created_path.unlink()
                except OSError:
                    pass
            raise
        for stale_path in previous_files:
            if stale_path in {output_path, metadata_path}:
                continue
            try:
                stale_path.unlink(missing_ok=True)
            except OSError:
                pass
        return record

    @_project_transaction
    def record_translation_failure(
        self, project_id: str, chapter_id: str, error: str
    ) -> None:
        manifest = self.load_project(project_id)
        chapter = next(
            (
                item
                for item in manifest.get("chapters", [])
                if item.get("id") == chapter_id
            ),
            None,
        )
        if not chapter:
            return
        translations = chapter.get("translations", [])
        if translations:
            chapter["status"] = (
                "需要复核"
                if translations[-1].get("review_status") == "需要复核"
                else "已翻译"
            )
            chapter["last_error"] = ""
            chapter["failed_at"] = ""
            manifest["updated_at"] = utc_now()
            _atomic_write_json(self.project_dir(project_id) / PROJECT_MANIFEST, manifest)
            return
        chapter["status"] = "翻译失败"
        chapter["last_error"] = str(error)[:1000]
        chapter["failed_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        _atomic_write_json(self.project_dir(project_id) / PROJECT_MANIFEST, manifest)

    @_project_transaction
    def record_residual_resolution_failure(
        self, project_id: str, chapter_id: str, residual_sources: Iterable[str]
    ) -> None:
        """Keep the last translation while surfacing exhausted residual retries."""
        manifest = self.load_project(project_id)
        chapter = next(
            (
                item for item in manifest.get("chapters", [])
                if item.get("id") == chapter_id
            ),
            None,
        )
        if not chapter:
            return
        sources = [str(item).strip() for item in residual_sources if str(item).strip()]
        now = utc_now()
        chapter["status"] = "翻译失败"
        chapter["last_error"] = (
            "残留原词在三次 API 润色后仍未解决：" + "、".join(sources)
        )[:1000]
        chapter["failed_at"] = now
        manifest["updated_at"] = now
        _atomic_write_json(self.project_dir(project_id) / PROJECT_MANIFEST, manifest)

    @_project_transaction
    def mark_chapters_retrying(
        self, project_id: str, chapter_ids: Iterable[str]
    ) -> dict:
        ids = set(chapter_ids)
        manifest = self.load_project(project_id)
        changed = False
        for chapter in manifest.get("chapters", []):
            if chapter.get("id") in ids and chapter.get("status") == "翻译失败":
                chapter["status"] = "重试中"
                chapter["last_error"] = ""
                changed = True
        if changed:
            manifest["updated_at"] = utc_now()
            _atomic_write_json(
                self.project_dir(project_id) / PROJECT_MANIFEST, manifest
            )
        return manifest

    @_project_transaction
    def mark_chapters_queued(
        self, project_id: str, chapter_ids: Iterable[str]
    ) -> dict:
        """Persist the visible batch queue state before workers begin."""
        ids = set(chapter_ids)
        manifest = self.load_project(project_id)
        changed = False
        for chapter in manifest.get("chapters", []):
            if chapter.get("id") in ids:
                chapter["status"] = "队列中"
                chapter["last_error"] = ""
                changed = True
        if changed:
            manifest["updated_at"] = utc_now()
            _atomic_write_json(
                self.project_dir(project_id) / PROJECT_MANIFEST, manifest
            )
        return manifest

    @_project_transaction
    def mark_chapter_processing(self, project_id: str, chapter_id: str) -> None:
        """Persist that a worker has started sending this chapter to the model."""
        manifest = self.load_project(project_id)
        chapter = next(
            (item for item in manifest.get("chapters", []) if item.get("id") == chapter_id),
            None,
        )
        if not chapter:
            return
        chapter["status"] = "处理中"
        chapter["last_error"] = ""
        manifest["updated_at"] = utc_now()
        _atomic_write_json(self.project_dir(project_id) / PROJECT_MANIFEST, manifest)

    @_project_transaction
    def restore_interrupted_chapters(
        self, project_id: str, chapter_ids: Iterable[str]
    ) -> dict:
        ids = set(chapter_ids)
        manifest = self.load_project(project_id)
        changed = False
        for chapter in manifest.get("chapters", []):
            if chapter.get("id") not in ids or chapter.get("status") not in {
                "队列中", "处理中", "重试中"
            }:
                continue
            translations = chapter.get("translations", [])
            if not translations:
                chapter["status"] = "待处理"
            elif translations[-1].get("review_status") == "需要复核":
                chapter["status"] = "需要复核"
            else:
                chapter["status"] = "已翻译"
            chapter["last_error"] = ""
            changed = True
        if changed:
            manifest["updated_at"] = utc_now()
            _atomic_write_json(
                self.project_dir(project_id) / PROJECT_MANIFEST, manifest
            )
        return manifest

    def read_latest_translation(
        self, project_id: str, chapter_id: str
    ) -> str | None:
        chapter = self.get_chapter(project_id, chapter_id)
        translations = chapter.get("translations", [])
        if not translations:
            return None
        path = (
            self.project_dir(project_id)
            / "output"
            / translations[-1].get("output_name", "")
        )
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    @_project_transaction
    def upsert_replacement(
        self,
        project_id: str,
        category: str,
        source: str,
        target: str,
        chapter_id: str | None = None,
        notes: str = "",
    ) -> dict:
        """按“原词”唯一写入；再次出现时复用同一条记录。"""
        source_value = source.strip()
        target_value = target.strip()
        if not source_value or not target_value:
            raise ValidationError("原词和替换词都不能为空。")
        rename_error = forced_rename_validation_error(category, source_value, target_value)
        if rename_error:
            raise ValidationError(rename_error)

        project_dir = self.project_dir(project_id)
        registry = self.load_registry(project_id, category)
        entries = registry.setdefault("entries", {})
        existing = entries.get(source_value)
        now = utc_now()
        if existing:
            if existing.get("target") != target_value:
                raise ValidationError(
                    f"“{source_value}”已经固定替换为“{existing.get('target', '')}”，"
                    "不能自动产生第二个名字。请通过人工确认流程更新。"
                )
            if chapter_id:
                chapters = existing.setdefault("source_chapters", [])
                if chapter_id not in chapters:
                    chapters.append(chapter_id)
            existing["last_seen_at"] = now
            record = existing
        else:
            record = {
                "source": source_value,
                "target": target_value,
                "status": "已确认",
                "notes": notes.strip(),
                "source_chapters": [chapter_id] if chapter_id else [],
                "created_at": now,
                "last_seen_at": now,
            }
            entries[source_value] = record
        registry["updated_at"] = now
        _atomic_write_json(self._registry_path(project_dir, category), registry)
        return record

    @_project_transaction
    def confirm_replacement(
        self, project_id: str, category: str, source: str, target: str,
        *, notes: str = "", allow_user_override: bool = False,
    ) -> dict:
        """Confirm a user-reviewed mapping, replacing only invalid identity history."""
        source_value = source.strip()
        target_value = target.strip()
        if not source_value or not target_value:
            raise ValidationError("原词和替换词都不能为空。")
        rename_error = forced_rename_validation_error(category, source_value, target_value)
        if rename_error:
            raise ValidationError(rename_error)
        registry = self.load_registry(project_id, category)
        existing = registry.setdefault("entries", {}).get(source_value)
        if (
            existing
            and existing.get("source") != existing.get("target")
            and not allow_user_override
        ):
            if existing.get("target") != target_value:
                raise ValidationError(
                    f"“{source_value}”已经固定替换为“{existing.get('target', '')}”。"
                )
            return existing
        now = utc_now()
        record = {
            "source": source_value,
            "target": target_value,
            "status": "已确认",
            "notes": notes.strip(),
            "source_chapters": (existing or {}).get("source_chapters", []),
            "created_at": (existing or {}).get("created_at", now),
            "last_seen_at": now,
        }
        registry["entries"][source_value] = record
        registry["updated_at"] = now
        _atomic_write_json(
            self._registry_path(self.project_dir(project_id), category), registry
        )
        return record

    def _create_registry(self, project_dir: Path, category: str) -> None:
        path = self._registry_path(project_dir, category)
        if path.exists():
            return
        now = utc_now()
        _atomic_write_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "category": category,
                "entries": {},
                "created_at": now,
                "updated_at": now,
            },
        )

    @staticmethod
    def _registry_path(project_dir: Path, category: str) -> Path:
        filename = _safe_filename(category, "其他") + ".json"
        return project_dir / "replacements" / filename

    def _copy_skills(
        self,
        project_dir: Path,
        skill_paths: Iterable[Path | str],
        *,
        skip_hashes: Iterable[str | None] = (),
    ) -> list[dict]:
        records: list[dict] = []
        known_hashes = {value for value in skip_hashes if value}
        skills_dir = project_dir / "skills"
        for raw_path in skill_paths:
            source_path = Path(raw_path)
            if not source_path.is_file():
                raise ValidationError(f"找不到规则/Skill 文件：{source_path}")
            data = source_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest in known_hashes:
                continue
            destination = self._available_destination(skills_dir, source_path.name)
            shutil.copy2(source_path, destination)
            records.append(
                {
                    "id": uuid.uuid4().hex,
                    "source_name": source_path.name,
                    "stored_name": destination.name,
                    "sha256": digest,
                    "added_at": utc_now(),
                }
            )
            known_hashes.add(digest)
        return records

    @staticmethod
    def _available_destination(folder: Path, original_name: str) -> Path:
        safe_name = _safe_filename(Path(original_name).name, "文件")
        candidate = folder / safe_name
        counter = 2
        while candidate.exists():
            candidate = folder / f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}"
            counter += 1
        return candidate


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class WindowsSecretStore:
    """使用 Windows DPAPI，将密钥绑定到当前 Windows 用户。"""

    prefix = "dpapi:"

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("当前系统不支持 Windows DPAPI。")
        self.crypt32 = ctypes.windll.crypt32
        self.kernel32 = ctypes.windll.kernel32

    @staticmethod
    def _blob(data: bytes):
        buffer = ctypes.create_string_buffer(data)
        blob = _DataBlob(
            len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        )
        return blob, buffer

    def protect(self, secret: str) -> str:
        if not secret:
            return ""
        source, source_buffer = self._blob(secret.encode("utf-8"))
        output = _DataBlob()
        if not self.crypt32.CryptProtectData(
            ctypes.byref(source),
            "NovelLocalizer API Key",
            None,
            None,
            None,
            0,
            ctypes.byref(output),
        ):
            raise ctypes.WinError()
        try:
            encrypted = ctypes.string_at(output.pbData, output.cbData)
            return self.prefix + base64.b64encode(encrypted).decode("ascii")
        finally:
            self.kernel32.LocalFree(output.pbData)
            del source_buffer

    def unprotect(self, encrypted: str) -> str:
        if not encrypted:
            return ""
        if not encrypted.startswith(self.prefix):
            raise ValidationError("API Key 的加密格式无法识别。")
        raw = base64.b64decode(encrypted[len(self.prefix) :])
        source, source_buffer = self._blob(raw)
        output = _DataBlob()
        if not self.crypt32.CryptUnprotectData(
            ctypes.byref(source),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(output),
        ):
            raise ctypes.WinError()
        try:
            return ctypes.string_at(output.pbData, output.cbData).decode("utf-8")
        finally:
            self.kernel32.LocalFree(output.pbData)
            del source_buffer


class APIConfigStore:
    def __init__(self, data_root: Path | str, secret_store=None):
        self.path = Path(data_root) / "config" / "api_profiles.json"
        self.secret_store = secret_store or WindowsSecretStore()

    def list_profiles(self) -> list[dict]:
        data = _read_json(self.path, {"schema_version": SCHEMA_VERSION, "profiles": []})
        profiles = data.get("profiles", []) if isinstance(data, dict) else []
        return sorted(profiles, key=lambda item: item.get("name", "").casefold())

    def get_profile(self, profile_id: str, reveal_key: bool = False) -> dict | None:
        for profile in self.list_profiles():
            if profile.get("id") == profile_id:
                result = dict(profile)
                if reveal_key:
                    result["api_key"] = self.secret_store.unprotect(
                        result.get("encrypted_api_key", "")
                    )
                result.pop("encrypted_api_key", None)
                return result
        return None

    def upsert_profile(
        self,
        *,
        profile_id: str | None,
        name: str,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: int,
    ) -> dict:
        clean_name = name.strip()
        if not clean_name:
            raise ValidationError("请输入配置名称。")
        if not provider.strip():
            raise ValidationError("请选择 API 服务商。")
        if not base_url.strip():
            raise ValidationError("请输入 API 地址。")
        if not model.strip():
            raise ValidationError("请输入模型名称。")
        if timeout_seconds < 10 or timeout_seconds > 600:
            raise ValidationError("请求超时时间应在 10～600 秒之间。")

        data = _read_json(
            self.path, {"schema_version": SCHEMA_VERSION, "profiles": []}
        )
        profiles = data.setdefault("profiles", [])
        existing = next(
            (item for item in profiles if item.get("id") == profile_id), None
        )
        now = utc_now()
        if existing is None:
            if not api_key:
                raise ValidationError("新建配置时请输入 API Key。")
            existing = {
                "id": uuid.uuid4().hex,
                "created_at": now,
                "encrypted_api_key": self.secret_store.protect(api_key),
            }
            profiles.append(existing)
        elif api_key:
            existing["encrypted_api_key"] = self.secret_store.protect(api_key)

        existing.update(
            {
                "name": clean_name,
                "provider": provider.strip(),
                "base_url": base_url.strip().rstrip("/"),
                "model": model.strip(),
                "timeout_seconds": timeout_seconds,
                "updated_at": now,
            }
        )
        _atomic_write_json(self.path, data)
        result = dict(existing)
        result.pop("encrypted_api_key", None)
        return result
