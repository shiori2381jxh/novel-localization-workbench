from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from novel_localizer.domain import ValidationError
from novel_localizer.storage import APIConfigStore, ProjectStore


class FakeSecretStore:
    def protect(self, secret: str) -> str:
        return "test:" + secret[::-1]

    def unprotect(self, encrypted: str) -> str:
        return encrypted.removeprefix("test:")[::-1]


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
            {"mode": "日语原文 → 改名润色", "max_output_tokens": 32000},
        )

        self.assertEqual(
            "日语原文 → 改名润色",
            saved["processing_settings"]["mode"],
        )
        self.assertEqual(32000, saved["processing_settings"]["max_output_tokens"])
        self.assertEqual({}, self.store.create_project(
            "小说B", "日语", "日语", ["人名"]
        ).get("processing_settings", {}))

    def test_processing_mode_is_locked_after_legacy_project_selects_it(self):
        project = self.store.create_project(
            "旧项目", "日语", "日语", ["人名"]
        )

        saved = self.store.set_processing_mode(
            project["id"], "日语原文 → 改名润色"
        )

        self.assertEqual("日语原文 → 改名润色", saved["processing_mode"])
        with self.assertRaisesRegex(ValidationError, "已锁定"):
            self.store.set_processing_mode(
                project["id"], "中文小说 → 日语本地化"
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
        self.store.upsert_replacement(
            project["id"], "人名", "セレン", "リゼ", chapters[0]["id"]
        )
        self.store.upsert_replacement(
            project["id"], "人名", "クラウス", "ユリウス", chapters[1]["id"]
        )

        rows = self.store.chapter_replacements(project["id"], chapters[0]["id"])

        self.assertEqual(["セレン"], [item["source"] for item in rows])
        self.assertEqual("人名", rows[0]["category"])

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
