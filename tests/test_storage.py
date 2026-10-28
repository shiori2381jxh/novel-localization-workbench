from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from novel_localizer.domain import ValidationError
from novel_localizer.storage import APIConfigStore, ProjectStore, _atomic_write_json


class FakeSecretStore:
    def protect(self, secret: str) -> str:
        return "test:" + secret[::-1]

    def unprotect(self, encrypted: str) -> str:
        return encrypted.removeprefix("test:")[::-1]


class AtomicWriteTests(unittest.TestCase):
    def test_atomic_json_write_retries_transient_permission_error(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "state.json"
            real_replace = os.replace
            attempts = 0

            def flaky_replace(source, target):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("temporarily locked")
                return real_replace(source, target)

            with patch("novel_localizer.storage.os.replace", side_effect=flaky_replace):
                _atomic_write_json(destination, {"saved": True})

            self.assertEqual(3, attempts)
            self.assertEqual({"saved": True}, json.loads(destination.read_text("utf-8")))

    def test_atomic_json_write_cleans_temp_after_permanent_permission_error(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            destination = folder / "state.json"

            with patch(
                "novel_localizer.storage.os.replace",
                side_effect=PermissionError("permanently locked"),
            ):
                with self.assertRaises(PermissionError):
                    _atomic_write_json(destination, {"saved": True})

            self.assertEqual([], list(folder.glob(".state.json.*.tmp")))


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ProjectStore(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_create_and_import_project(self):
        source = self.root / "第一话.txt"
        source.write_text("赵一去了上海。", encoding="utf-8")
        project = self.store.create_project(
            "小说A", "中文（简体）", "日语", ["人名", "地名"]
        )

        result = self.store.import_chapters(project["id"], [source])
        reimport = self.store.import_chapters(project["id"], [source])
        manifest = self.store.load_project(project["id"])

        self.assertEqual(1, len(result.imported))
        self.assertEqual(0, len(reimport.imported))
        self.assertEqual(1, len(reimport.skipped))
        self.assertEqual(1, len(manifest["chapters"]))
        self.assertTrue(
            (self.store.project_dir(project["id"]) / "replacements" / "人名.json").exists()
        )

    def test_concurrent_chapter_status_updates_do_not_overwrite_each_other(self):
        first_path = self.root / "一.txt"
        second_path = self.root / "二.txt"
        first_path.write_text("一", encoding="utf-8")
        second_path.write_text("二", encoding="utf-8")
        project = self.store.create_project("并发项目", "中文", "日语", ["人名"])
        chapters = self.store.import_chapters(
            project["id"], [first_path, second_path]
        ).imported
        self.store.mark_chapters_queued(
            project["id"], [chapter["id"] for chapter in chapters]
        )
        original_load = self.store.load_project
        barrier = threading.Barrier(2)

        def synchronized_load(project_id):
            manifest = original_load(project_id)
            try:
                barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
            return manifest

        self.store.load_project = synchronized_load
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        self.store.mark_chapter_processing,
                        project["id"],
                        chapter["id"],
                    )
                    for chapter in chapters
                ]
                for future in futures:
                    future.result()
        finally:
            self.store.load_project = original_load

        statuses = {
            chapter["id"]: self.store.get_chapter(project["id"], chapter["id"])["status"]
            for chapter in chapters
        }
        self.assertEqual({chapter["id"]: "处理中" for chapter in chapters}, statuses)

    def test_concurrent_replacement_updates_do_not_overwrite_each_other(self):
        project = self.store.create_project("并发替换", "中文", "日语", ["术语"])
        original_load = self.store.load_registry
        barrier = threading.Barrier(2)

        def synchronized_load(project_id, category):
            registry = original_load(project_id, category)
            try:
                barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
            return registry

        self.store.load_registry = synchronized_load
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        self.store.upsert_replacement,
                        project["id"], "术语", source, target,
                    )
                    for source, target in (("甲", "A"), ("乙", "B"))
                ]
                for future in futures:
                    future.result()
        finally:
            self.store.load_registry = original_load

        entries = self.store.load_registry(project["id"], "术语")["entries"]
        self.assertEqual({"甲", "乙"}, set(entries))

    def test_replacement_is_unique_by_source(self):
        project = self.store.create_project(
            "小说A", "中文（简体）", "日语", ["人名"]
        )
        first = self.store.upsert_replacement(
            project["id"], "人名", "张三", "夏目 二郎", "chapter-1"
        )
        again = self.store.upsert_replacement(
            project["id"], "人名", "张三", "夏目 二郎", "chapter-2"
        )

        self.assertEqual(first["target"], again["target"])
        self.assertEqual(["chapter-1", "chapter-2"], again["source_chapters"])
        with self.assertRaises(ValidationError):
            self.store.upsert_replacement(
                project["id"], "人名", "张三", "田中 太郎", "chapter-3"
            )

        registry = self.store.load_registry(project["id"], "人名")
        self.assertEqual(1, len(registry["entries"]))

    def test_forced_rename_categories_reject_identity_mapping(self):
        project = self.store.create_project(
            "小说A", "日语", "日语", ["人名", "第一人称"]
        )

        with self.assertRaisesRegex(ValidationError, "必须使用不同的新名称"):
            self.store.upsert_replacement(
                project["id"], "人名", "セレン", "セレン"
            )

        record = self.store.upsert_replacement(
            project["id"], "第一人称", "私", "私"
        )
        self.assertEqual("私", record["target"])

    def test_forced_rename_traditional_category_alias_rejects_identity_mapping(self):
        project = self.store.create_project(
            "小说A", "中文", "中文", ["種族名"]
        )

        with self.assertRaisesRegex(ValidationError, "必须使用不同的新名称"):
            self.store.upsert_replacement(
                project["id"], "種族名", "精灵", "精灵"
            )

    def test_forced_rename_rejects_same_initial_japanese_name(self):
        project = self.store.create_project(
            "小说A", "日语", "日语", ["人名"]
        )

        with self.assertRaisesRegex(ValidationError, "首音"):
            self.store.upsert_replacement(
                project["id"], "人名", "アゼット", "アリシア"
            )

    def test_forced_rename_accepts_distinct_japanese_name(self):
        project = self.store.create_project(
            "小说A", "日语", "日语", ["人名"]
        )

        record = self.store.upsert_replacement(
            project["id"], "人名", "アゼット", "ミサキ"
        )

        self.assertEqual("ミサキ", record["target"])

    def test_processing_settings_are_saved_per_project(self):
        project = self.store.create_project(
            "小说A", "日语", "日语", ["人名"]
        )

        saved = self.store.update_processing_settings(
            project["id"],
            {"mode": "润色", "max_output_tokens": 32000},
        )

        self.assertEqual(
            "润色",
            saved["processing_settings"]["mode"],
        )
        self.assertEqual(32000, saved["processing_settings"]["max_output_tokens"])
        self.assertEqual({}, self.store.create_project(
            "小说B", "日语", "日语", ["人名"]
        ).get("processing_settings", {}))

    def test_new_project_defaults_to_proper_noun_localization_enabled(self):
        project = self.store.create_project(
            "默认改名", "中文（简体）", "中文（繁体）", ["人名"]
        )

        self.assertTrue(project["proper_noun_localization"])
        self.assertTrue(
            self.store.proper_noun_localization_enabled(project["id"])
        )

        manifest_path = self.store.project_dir(project["id"]) / "project.json"
        legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy.pop("proper_noun_localization")
        _atomic_write_json(manifest_path, legacy)

        self.assertTrue(
            self.store.proper_noun_localization_enabled(project["id"])
        )

    def test_project_rename_setting_can_be_disabled_and_updated(self):
        project = self.store.create_project(
            "关闭改名", "中文（简体）", "中文（繁体）", ["人名"],
            proper_noun_localization=False,
        )

        self.assertFalse(project["proper_noun_localization"])
        self.assertFalse(
            self.store.proper_noun_localization_enabled(project["id"])
        )

        updated = self.store.update_project_settings(
            project["id"], proper_noun_localization=True
        )

        self.assertTrue(updated["proper_noun_localization"])
        self.assertTrue(
            self.store.load_project(project["id"])["proper_noun_localization"]
        )

    def test_processing_mode_is_locked_after_legacy_project_selects_it(self):
        project = self.store.create_project(
            "旧项目", "日语", "日语", ["人名"]
        )

        saved = self.store.set_processing_mode(
            project["id"], "润色"
        )

        self.assertEqual("润色", saved["processing_mode"])
        with self.assertRaisesRegex(ValidationError, "已锁定"):
            self.store.set_processing_mode(
                project["id"], "翻译"
            )

    def test_failed_chapter_changes_to_retrying_before_another_attempt(self):
        project = self.store.create_project(
            "小说A", "日语", "日语", ["人名"]
        )
        chapter = self.store.import_chapter_text(project["id"], "正文").imported[0]
        self.store.record_translation_failure(project["id"], chapter["id"], "旧错误")

        self.store.mark_chapters_retrying(project["id"], [chapter["id"]])

        saved = self.store.get_chapter(project["id"], chapter["id"])
        self.assertEqual("重试中", saved["status"])
        self.assertEqual("", saved["last_error"])

    def test_residual_resolution_failure_preserves_latest_translation(self):
        project = self.store.create_project("小说A", "日语", "日语", ["人名"])
        chapter = self.store.import_chapter_text(project["id"], "正文").imported[0]
        record = self.store.save_translation(
            project["id"], chapter["id"], "最后译文",
            profile_id="p", provider="test", model="m",
            review_status="需要复核", residual_sources=["莉娜"],
        )

        self.store.record_residual_resolution_failure(
            project["id"], chapter["id"], ["莉娜"]
        )

        saved = self.store.get_chapter(project["id"], chapter["id"])
        self.assertEqual("翻译失败", saved["status"])
        self.assertIn("三次 API 润色后", saved["last_error"])
        self.assertEqual(record["output_name"], saved["translations"][-1]["output_name"])
        self.assertEqual(
            "最后译文\n",
            self.store.read_latest_translation(project["id"], chapter["id"]),
        )

    def test_restore_interrupted_chapter_without_translation_to_pending(self):
        source = self.root / "未完成.txt"
        source.write_text("正文", encoding="utf-8")
        project = self.store.create_project("取消恢复", "中文", "日语", ["人名"])
        chapter = self.store.import_chapters(project["id"], [source]).imported[0]
        self.store.mark_chapters_queued(project["id"], [chapter["id"]])
        self.store.mark_chapter_processing(project["id"], chapter["id"])

        self.store.restore_interrupted_chapters(project["id"], [chapter["id"]])

        restored = self.store.get_chapter(project["id"], chapter["id"])
        self.assertEqual("待处理", restored["status"])

    def test_restore_interrupted_chapters_preserves_completed_translation_state(self):
        complete_path = self.root / "完成.txt"
        review_path = self.root / "复核.txt"
        complete_path.write_text("甲", encoding="utf-8")
        review_path.write_text("乙", encoding="utf-8")
        project = self.store.create_project("取消恢复", "中文", "日语", ["人名"])
        complete, review = self.store.import_chapters(
            project["id"], [complete_path, review_path]
        ).imported
        self.store.save_translation(
            project["id"], complete["id"], "A",
            profile_id="p", provider="test", model="m", review_status="完成",
        )
        self.store.save_translation(
            project["id"], review["id"], "B",
            profile_id="p", provider="test", model="m", review_status="需要复核",
            residual_sources=["乙"],
        )
        self.store.mark_chapters_queued(
            project["id"], [complete["id"], review["id"]]
        )

        self.store.restore_interrupted_chapters(
            project["id"], [complete["id"], review["id"]]
        )

        self.assertEqual(
            "已翻译", self.store.get_chapter(project["id"], complete["id"])["status"]
        )
        self.assertEqual(
            "需要复核", self.store.get_chapter(project["id"], review["id"])["status"]
        )

    def test_translation_checkpoint_round_trips_and_can_be_cleared(self):
        project = self.store.create_project(
            "小说A", "日语", "日语", ["人名"]
        )
        chapter = self.store.import_chapter_text(project["id"], "正文").imported[0]
        checkpoint = {
            "source_hash": "abc",
            "completed_segments": ["第一段译文"],
        }

        self.store.save_translation_checkpoint(
            project["id"], chapter["id"], checkpoint
        )

        self.assertEqual(
            checkpoint,
            self.store.load_translation_checkpoint(project["id"], chapter["id"]),
        )
        self.store.clear_translation_checkpoint(project["id"], chapter["id"])
        self.assertIsNone(
            self.store.load_translation_checkpoint(project["id"], chapter["id"])
        )

    def test_translation_save_removes_new_outputs_when_manifest_write_fails(self):
        source = self.root / "失败回滚.txt"
        source.write_text("正文", encoding="utf-8")
        project = self.store.create_project("回滚项目", "中文", "日语", ["人名"])
        chapter = self.store.import_chapters(project["id"], [source]).imported[0]
        output_dir = self.store.project_dir(project["id"]) / "output"
        existing = output_dir / "既有文件.txt"
        existing.write_text("保留", encoding="utf-8")
        json_writes = 0

        def fail_manifest_write(path, data):
            nonlocal json_writes
            json_writes += 1
            if json_writes == 2:
                raise PermissionError("manifest remains locked")
            return _atomic_write_json(path, data)

        with patch(
            "novel_localizer.storage._atomic_write_json",
            side_effect=fail_manifest_write,
        ):
            with self.assertRaises(PermissionError):
                self.store.save_translation(
                    project["id"], chapter["id"], "译文",
                    profile_id="p", provider="test", model="m",
                )

        self.assertEqual(["既有文件.txt"], [item.name for item in output_dir.iterdir()])
        self.assertEqual(
            [],
            self.store.get_chapter(project["id"], chapter["id"]).get("translations", []),
        )

    def test_translation_save_replaces_the_previous_output_for_the_same_chapter(self):
        source = self.root / "重做覆盖.txt"
        source.write_text("正文", encoding="utf-8")
        project = self.store.create_project("覆盖项目", "中文", "日语", ["人名"])
        chapter = self.store.import_chapters(project["id"], [source]).imported[0]

        first = self.store.save_translation(
            project["id"], chapter["id"], "旧译文",
            profile_id="p", provider="test", model="m",
        )
        second = self.store.save_translation(
            project["id"], chapter["id"], "新译文",
            profile_id="p", provider="test", model="m",
        )

        output_dir = self.store.project_dir(project["id"]) / "output"
        current = self.store.get_chapter(project["id"], chapter["id"])
        self.assertEqual(first["output_name"], second["output_name"])
        self.assertEqual("新译文\n", (output_dir / second["output_name"]).read_text(encoding="utf-8"))
        self.assertEqual([second["id"]], [item["id"] for item in current["translations"]])
        self.assertEqual(
            sorted([second["output_name"], second["metadata_name"]]),
            sorted(item.name for item in output_dir.iterdir()),
        )

    def test_completed_output_contains_only_currently_completed_translation_files(self):
        project = self.store.create_project("完成译文项目", "中文", "日语", ["人名"])
        done = self.store.import_chapter_text(project["id"], "完成章节").imported[0]
        review = self.store.import_chapter_text(project["id"], "复核章节").imported[0]
        done_record = self.store.save_translation(
            project["id"], done["id"], "完成译文",
            profile_id="p", provider="test", model="m",
        )
        review_record = self.store.save_translation(
            project["id"], review["id"], "复核译文",
            profile_id="p", provider="test", model="m",
            review_status="需要复核", residual_sources=["残留名"],
        )

        completed_dir = self.store.refresh_completed_output(project["id"])
        user_note = completed_dir / "用户备注.txt"
        user_note.write_text("不要删除", encoding="utf-8")

        self.assertEqual(
            "完成译文\n",
            (completed_dir / done_record["output_name"]).read_text(encoding="utf-8"),
        )
        self.assertFalse((completed_dir / review_record["output_name"]).exists())

        self.store.save_translation(
            project["id"], done["id"], "改为待复核",
            profile_id="p", provider="test", model="m",
            review_status="需要复核", residual_sources=["另一个残留名"],
        )
        self.store.refresh_completed_output(project["id"])

        self.assertFalse((completed_dir / done_record["output_name"]).exists())
        self.assertEqual("不要删除", user_note.read_text(encoding="utf-8"))

    def test_completed_output_recovers_from_a_malformed_generated_index(self):
        project = self.store.create_project("损坏索引项目", "中文", "日语", ["人名"])
        chapter = self.store.import_chapter_text(project["id"], "正文").imported[0]
        record = self.store.save_translation(
            project["id"], chapter["id"], "完成译文",
            profile_id="p", provider="test", model="m",
        )
        completed_dir = self.store.project_dir(project["id"]) / "已完成译文"
        completed_dir.mkdir()
        (completed_dir / ".novel_localizer_completed_output.json").write_text(
            "[]", encoding="utf-8"
        )

        refreshed = self.store.refresh_completed_output(project["id"])

        self.assertTrue((refreshed / record["output_name"]).is_file())

    def test_translation_failure_keeps_the_status_of_an_existing_translation(self):
        project = self.store.create_project("恢复项目", "中文", "日语", ["人名"])
        chapter = self.store.import_chapter_text(project["id"], "正文").imported[0]
        self.store.save_translation(
            project["id"], chapter["id"], "已有译文",
            profile_id="p", provider="test", model="m",
            review_status="需要复核", residual_sources=["残留名"],
        )

        self.store.record_translation_failure(project["id"], chapter["id"], "重做 API 失败")

        restored = self.store.get_chapter(project["id"], chapter["id"])
        self.assertEqual("需要复核", restored["status"])
        self.assertEqual("", restored["last_error"])
        self.assertEqual("已有译文\n", self.store.read_latest_translation(project["id"], chapter["id"]))

    def test_custom_category_can_be_added(self):
        project = self.store.create_project(
            "医院题材", "中文（简体）", "日语", ["人名", "地名"]
        )
        added = self.store.ensure_categories(project["id"], ["医院名", "食物名"])
        manifest = self.store.load_project(project["id"])
        self.assertEqual(["医院名", "食物名"], added)
        self.assertIn("医院名", manifest["categories"])

    def test_chapter_replacements_returns_only_selected_chapter_terms(self):
        project = self.store.create_project(
            "小说", "日语", "日语", ["人名", "地名"]
        )
        first = self.root / "一.txt"
        second = self.root / "二.txt"
        first.write_text("セレンは王都へ行った。", encoding="utf-8")
        second.write_text("クラウスは村へ行った。", encoding="utf-8")
        chapters = self.store.import_chapters(project["id"], [first, second]).imported
        self.store.upsert_chapter_mapping(
            project["id"], chapters[0]["id"], "人名", "セレン", "リゼ"
        )
        self.store.upsert_chapter_mapping(
            project["id"], chapters[1]["id"], "人名", "クラウス", "ユリウス"
        )

        rows = self.store.chapter_replacements(project["id"], chapters[0]["id"])

        self.assertEqual(["セレン"], [item["source"] for item in rows])
        self.assertEqual("人名", rows[0]["category"])

    def test_chapter_replacements_includes_local_names_without_global_registry_entry(self):
        project = self.store.create_project("小说", "日语", "日语", ["人名"])
        source = self.root / "一.txt"
        source.write_text("セレン。", encoding="utf-8")
        chapter = self.store.import_chapters(project["id"], [source]).imported[0]
        self.store.upsert_chapter_mapping(
            project["id"], chapter["id"], "人名", "セレン", "リゼ"
        )

        rows = self.store.chapter_replacements(project["id"], chapter["id"])

        self.assertEqual(["セレン"], [item["source"] for item in rows])
        self.assertEqual("リゼ", rows[0]["target"])

    def test_chapter_mappings_are_isolated_between_independent_stories(self):
        project = self.store.create_project("短篇", "日语", "中文", ["人名"])
        first = self.root / "一.txt"
        second = self.root / "二.txt"
        first.write_text("アリス。", encoding="utf-8")
        second.write_text("アリス。第二篇。", encoding="utf-8")
        chapters = self.store.import_chapters(project["id"], [first, second]).imported

        self.store.upsert_chapter_mapping(
            project["id"], chapters[0]["id"], "人名", "アリス", "林爱丽"
        )
        self.store.upsert_chapter_mapping(
            project["id"], chapters[1]["id"], "人名", "アリス", "苏爱丽"
        )

        self.assertEqual(
            "林爱丽",
            self.store.chapter_mappings(project["id"], chapters[0]["id"])["人名"]["アリス"]["target"],
        )
        self.assertEqual(
            "苏爱丽",
            self.store.chapter_mappings(project["id"], chapters[1]["id"])["人名"]["アリス"]["target"],
        )

    def test_explicit_empty_translation_is_saved_for_metadata_only_file(self):
        project = self.store.create_project("小说", "日语", "日语", ["人名"])
        chapter = self.store.import_chapter_text(
            project["id"], "作者后记"
        ).imported[0]

        record = self.store.save_translation(
            project["id"], chapter["id"], "",
            profile_id="p", provider="test", model="m", allow_empty=True,
        )

        self.assertEqual("", self.store.read_latest_translation(project["id"], chapter["id"]).strip())
        self.assertEqual("已翻译", self.store.get_chapter(project["id"], chapter["id"])["status"])
        self.assertTrue(record["output_name"].endswith("_译文.txt"))

    def test_delete_chapters_removes_files_outputs_and_replacement_references(self):
        project = self.store.create_project(
            "小说", "日语", "日语", ["人名"]
        )
        first = self.root / "一.txt"
        second = self.root / "二.txt"
        first.write_text("セレン。", encoding="utf-8")
        second.write_text("セレン。別章。", encoding="utf-8")
        chapters = self.store.import_chapters(project["id"], [first, second]).imported
        self.store.upsert_replacement(
            project["id"], "人名", "セレン", "リゼ", chapters[0]["id"]
        )
        self.store.upsert_replacement(
            project["id"], "人名", "セレン", "リゼ", chapters[1]["id"]
        )
        record = self.store.save_translation(
            project["id"], chapters[0]["id"], "リゼ。",
            profile_id="p", provider="test", model="m",
        )
        source_path = self.store.project_dir(project["id"]) / "source" / chapters[0]["stored_name"]
        output_path = self.store.project_dir(project["id"]) / "output" / record["output_name"]
        metadata_path = self.store.project_dir(project["id"]) / "output" / record["metadata_name"]

        deleted = self.store.delete_chapters(project["id"], [chapters[0]["id"]])

        self.assertEqual([chapters[0]["id"]], [item["id"] for item in deleted])
        self.assertFalse(source_path.exists())
        self.assertFalse(output_path.exists())
        self.assertFalse(metadata_path.exists())
        manifest = self.store.load_project(project["id"])
        self.assertEqual([chapters[1]["id"]], [item["id"] for item in manifest["chapters"]])
        entry = self.store.load_registry(project["id"], "人名")["entries"]["セレン"]
        self.assertEqual([chapters[1]["id"]], entry["source_chapters"])

    def test_same_skill_is_not_copied_twice(self):
        skill = self.root / "医院题材.md"
        skill.write_text("记录医院名和食物名", encoding="utf-8")
        project = self.store.create_project(
            "医院题材", "中文（简体）", "日语", ["人名"], [skill]
        )

        added = self.store.add_skills(project["id"], [skill])
        manifest = self.store.load_project(project["id"])
        copied = list((self.store.project_dir(project["id"]) / "skills").iterdir())

        self.assertEqual([], added)
        self.assertEqual(1, len(manifest["skills"]))
        self.assertEqual(1, len(copied))

    def test_pasted_chapter_is_saved_as_one_named_chapter_and_deduplicated(self):
        project = self.store.create_project(
            "粘贴项目", "中文（简体）", "日语", ["人名"]
        )
        first = self.store.import_chapter_text(project["id"], "  第一段。\n\n第二段。  ")
        repeated = self.store.import_chapter_text(project["id"], "第一段。\n\n第二段。")
        saved = self.store.read_chapter_text(project["id"], first.imported[0]["id"])

        self.assertEqual(1, len(first.imported))
        self.assertEqual("粘贴章节 1.txt", first.imported[0]["source_name"])
        self.assertEqual("第一段。\n\n第二段。", saved)
        self.assertEqual(1, len(repeated.skipped))

    def test_pasted_skill_is_saved_and_deduplicated(self):
        project = self.store.create_project(
            "粘贴项目", "中文（简体）", "日语", ["人名"]
        )
        first = self.store.add_skill_text(project["id"], "  女主角使用敬语。  ")
        repeated = self.store.add_skill_text(project["id"], "女主角使用敬语。")
        skills = self.store.read_project_skills(project["id"])

        self.assertEqual(1, len(first))
        self.assertEqual("粘贴规则 1.md", first[0]["source_name"])
        self.assertEqual([], repeated)
        self.assertEqual("女主角使用敬语。", skills[0]["content"])

    def test_skill_analysis_is_reused_only_for_the_matching_content_hash(self):
        project = self.store.create_project(
            "规则缓存项目", "中文（简体）", "中文（繁体）", ["人名"],
            processing_mode="润色",
        )
        skill = self.store.add_skill_text(project["id"], "将王城改为新名称。 ")[0]

        self.assertEqual(
            [skill["id"]],
            [item["id"] for item in self.store.pending_skill_analysis(project["id"])],
        )
        self.store.save_skill_analysis(
            project["id"], skill["id"], skill["sha256"],
            {"replacement_rules": [{"category": "地名", "action": "replace"}]},
        )
        self.assertEqual([], self.store.pending_skill_analysis(project["id"]))

        self.store.update_skill(project["id"], skill["id"], "保留王城名称。")
        self.assertEqual(
            [skill["id"]],
            [item["id"] for item in self.store.pending_skill_analysis(project["id"])],
        )

    def test_skill_can_be_updated_by_id(self):
        project = self.store.create_project(
            "改规则项目", "中文（简体）", "日语", ["人名"]
        )
        created = self.store.add_skill_text(project["id"], "旧规则。")

        updated = self.store.update_skill(
            project["id"], created[0]["id"], "  新规则。  "
        )
        skills = self.store.read_project_skills(project["id"])

        self.assertEqual(created[0]["id"], updated["id"])
        self.assertNotEqual(created[0]["sha256"], updated["sha256"])
        self.assertEqual("新规则。", skills[0]["content"])
        self.assertEqual(created[0]["id"], skills[0]["id"])

    def test_skill_update_rejects_empty_and_duplicate_content(self):
        project = self.store.create_project(
            "规则校验项目", "中文（简体）", "日语", ["人名"]
        )
        first = self.store.add_skill_text(project["id"], "规则一。")
        second = self.store.add_skill_text(project["id"], "规则二。")

        with self.assertRaises(ValidationError):
            self.store.update_skill(project["id"], first[0]["id"], "  ")
        with self.assertRaises(ValidationError):
            self.store.update_skill(
                project["id"], first[0]["id"], "规则二。"
            )

        self.assertEqual(
            ["规则一。", "规则二。"],
            [item["content"] for item in self.store.read_project_skills(project["id"])],
        )

    def test_multiple_skills_can_be_deleted_even_if_a_file_is_missing(self):
        project = self.store.create_project(
            "删除规则项目", "中文（简体）", "日语", ["人名"]
        )
        first = self.store.add_skill_text(project["id"], "规则一。")[0]
        second = self.store.add_skill_text(project["id"], "规则二。")[0]
        third = self.store.add_skill_text(project["id"], "规则三。")[0]
        missing_path = self.store.project_dir(project["id"]) / "skills" / second["stored_name"]
        missing_path.unlink()

        deleted = self.store.delete_skills(
            project["id"], [first["id"], second["id"], "unknown"]
        )
        manifest = self.store.load_project(project["id"])

        self.assertEqual([first["id"], second["id"]], [item["id"] for item in deleted])
        self.assertEqual([third["id"]], [item["id"] for item in manifest["skills"]])
        self.assertFalse(
            (self.store.project_dir(project["id"]) / "skills" / first["stored_name"]).exists()
        )


class APIConfigStoreTests(unittest.TestCase):
    def test_key_is_not_stored_as_plaintext(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = APIConfigStore(root, secret_store=FakeSecretStore())
            saved = store.upsert_profile(
                profile_id=None,
                name="默认配置",
                provider="OpenAI",
                base_url="https://api.openai.com/v1",
                model="example-model",
                api_key="secret-value",
                timeout_seconds=120,
            )
            raw = (root / "config" / "api_profiles.json").read_text("utf-8")
            loaded = store.get_profile(saved["id"], reveal_key=True)

            self.assertNotIn("secret-value", raw)
            self.assertEqual("secret-value", loaded["api_key"])
            self.assertEqual("example-model", loaded["model"])
            json.loads(raw)


if __name__ == "__main__":
    unittest.main()
