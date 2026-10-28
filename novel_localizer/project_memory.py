from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from functools import wraps

from .domain import ValidationError, forced_rename_validation_error
from .storage import ProjectStore, _atomic_write_json, _read_json, utc_now


def _project_memory_transaction(method):
    @wraps(method)
    def locked(self, project_id, *args, **kwargs):
        with self.store.project_lock(project_id):
            return method(self, project_id, *args, **kwargs)

    return locked


class ProjectMemoryStore:
    """Durable, user-confirmed project facts used across model calls."""

    def __init__(self, store: ProjectStore):
        self.store = store

    def _path(self, project_id: str):
        return self.store.project_dir(project_id) / "memory" / "project_memory.json"

    def load(self, project_id: str) -> dict:
        return _read_json(
            self._path(project_id),
            {
                "schema_version": 1,
                "project_id": project_id,
                "version": 0,
                "entities": [],
                "chapter_memory": {},
                "preparation": {},
                "updated_at": "",
            },
        )

    @_project_memory_transaction
    def save_candidates(
        self, project_id: str, chapter_ids: list[str], candidates: list[dict],
        *, chapter_memory: dict | None = None,
        preparation: dict | None = None,
    ) -> dict:
        memory = self.load(project_id)
        by_key = {
            (str(item.get("category", "")), str(item.get("source", ""))): item
            for item in memory.get("entities", [])
        }
        for raw in candidates:
            category = str(raw.get("category", "")).strip()
            source = str(raw.get("source", "")).strip()
            target = str(raw.get("target", "")).strip()
            if not category or not source:
                continue
            existing = by_key.get((category, source))
            if existing and existing.get("status") == "已确认":
                continue
            by_key[(category, source)] = {
                "category": category,
                "source": source,
                "target": target,
                "notes": str(raw.get("notes", "")).strip(),
                "details": deepcopy(raw.get("details", {})),
                "status": "待确认",
                "source_chapters": list(dict.fromkeys(chapter_ids)),
            }
        memory["entities"] = list(by_key.values())
        if chapter_memory is not None:
            memory["draft_chapter_memory"] = deepcopy(chapter_memory)
        if preparation is not None:
            memory["draft_preparation"] = deepcopy(preparation)
        return self._write(memory)

    @_project_memory_transaction
    def confirm_candidates(self, project_id: str, edits: list[dict]) -> dict:
        memory = self.load(project_id)
        entities = {
            (str(item.get("category", "")), str(item.get("source", ""))): item
            for item in memory.get("entities", [])
        }
        normalized_edits: list[dict] = []
        for edit in edits:
            category = str(edit.get("category", "")).strip()
            source = str(edit.get("source", "")).strip()
            target = str(edit.get("target", "")).strip()
            if not category or not source or not target:
                raise ValidationError("类别、原词和新名称都不能为空。")
            rename_error = forced_rename_validation_error(category, source, target)
            if rename_error:
                raise ValidationError(rename_error)
            normalized_edits.append(
                {
                    "category": category,
                    "source": source,
                    "target": target,
                    "notes": str(edit.get("notes", "")).strip(),
                    "details": deepcopy(edit.get("details", {})),
                }
            )
        self.store.ensure_categories(
            project_id,
            [item["category"] for item in normalized_edits],
        )
        for edit in normalized_edits:
            category = edit["category"]
            source = edit["source"]
            target = edit["target"]
            current = entities.get((category, source), {})
            current.update(
                {
                    "category": category,
                    "source": source,
                    "target": target,
                    "notes": edit.get("notes") or current.get("notes", ""),
                    "details": deepcopy(edit.get("details", current.get("details", {}))),
                    "status": "已确认",
                    "source_chapters": current.get("source_chapters", []),
                }
            )
            entities[(category, source)] = current
            self.store.confirm_replacement(
                project_id, category, source, target,
                notes=current.get("notes", ""),
                allow_user_override=True,
            )
        memory["entities"] = list(entities.values())
        if "draft_chapter_memory" in memory:
            memory["chapter_memory"] = memory.pop("draft_chapter_memory")
        if "draft_preparation" in memory:
            memory["preparation"] = memory.pop("draft_preparation")
        return self._write(memory)

    def confirmed_entities(self, project_id: str) -> list[dict]:
        return [
            deepcopy(item) for item in self.load(project_id).get("entities", [])
            if item.get("status") == "已确认"
        ]

    @_project_memory_transaction
    def update_chapter_memory(
        self, project_id: str, chapter_id: str, chapter_memory: dict
    ) -> dict:
        memory = self.load(project_id)
        all_chapters = memory.setdefault("chapter_memory", {})
        all_chapters[chapter_id] = deepcopy(chapter_memory)
        return self._write(memory)

    def preparation_snapshot(
        self, project_id: str, chapter_ids: list[str], mode: str
    ) -> dict:
        project = self.store.load_project(project_id)
        chapters = {item["id"]: item for item in project.get("chapters", [])}
        skills = self.store.read_project_skills(project_id)
        skill_payload = [
            {"name": item["name"], "content": item["content"]} for item in skills
        ]
        skill_hash = hashlib.sha256(
            json.dumps(skill_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "mode": mode,
            "chapters": {
                chapter_id: chapters.get(chapter_id, {}).get("sha256", "")
                for chapter_id in chapter_ids
            },
            "skill_hash": skill_hash,
            "skill_files": [item["name"] for item in skills],
        }

    def is_prepared(
        self, project_id: str, chapter_ids: list[str], mode: str
    ) -> bool:
        memory = self.load(project_id)
        if any(item.get("status") != "已确认" for item in memory.get("entities", [])):
            return False
        expected = self.preparation_snapshot(project_id, chapter_ids, mode)
        actual = memory.get("preparation", {})
        return (
            actual.get("mode") == expected["mode"]
            and actual.get("skill_hash") == expected["skill_hash"]
            and all(
                actual.get("chapters", {}).get(chapter_id) == source_hash
                for chapter_id, source_hash in expected["chapters"].items()
            )
        )

    def _write(self, memory: dict) -> dict:
        memory["version"] = int(memory.get("version", 0)) + 1
        memory["updated_at"] = utc_now()
        _atomic_write_json(self._path(memory["project_id"]), memory)
        return deepcopy(memory)
