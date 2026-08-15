from __future__ import annotations

import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QAbstractItemView,
    QTableWidgetItem,
)

from novel_localizer.desktop_app import (
    FileListBox, ProjectImportTab, ResidualReviewDialog, TranslationTab,
    available_name_candidates, split_review_edits,
)
from novel_localizer.storage import APIConfigStore, ProjectStore
from novel_localizer.translator import format_batch_start_log


class DesktopLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_batch_log_identifies_configuration_without_api_key(self):
        profile = {
            "provider": "OpenAI 兼容接口",
            "base_url": "https://example.test/v1?api_key=do-not-log",
            "model": "example-model",
            "api_key": "secret-key",
            "timeout_seconds": 90,
        }

        log = format_batch_start_log(
            profile,
            chapter_count=8,
            mode="本地化改写",
            max_output_tokens=12000,
        )

        self.assertIn("共 8 章", log)
        self.assertIn("OpenAI 兼容接口", log)
        self.assertIn("example-model", log)
        self.assertIn("本地化改写", log)
        self.assertIn("12000", log)
        self.assertIn("https://example.test/v1", log)
        self.assertNotIn("do-not-log", log)
        self.assertNotIn("secret-key", log)

    def test_import_panel_accepts_direct_text(self):
        panel = FileListBox("规则 / Skill", "可粘贴", "选择规则文件", "*.md")
        panel.text_edit.setPlainText("女主角使用敬语。")
        self.assertEqual("女主角使用敬语。", panel.text())

    def test_translation_tab_uses_explicit_modes_and_one_click_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            api_store = APIConfigStore(root / "api.json")
            tab = TranslationTab(store, api_store)

            self.assertFalse(tab.mode_display.isEnabled())
            self.assertEqual("开始处理", tab.start_button.text())
            self.assertFalse(hasattr(tab, "prepare_button"))

    def test_name_candidates_exclude_targets_already_used_by_the_project(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "name_pools.json").write_text(
                '{"japanese_male":["已用名","可选名"]}', encoding="utf-8"
            )
            store = ProjectStore(root)
            project = store.create_project("项目", "日语", "日语", ["人名"])
            store.confirm_replacement(
                project["id"], "人名", "原名", "已用名", allow_user_override=True
            )
            self.assertEqual(["可选名"], available_name_candidates(store, project["id"]))

    def test_residual_review_dialog_defaults_to_ignore(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project("项目", "日语", "日语", ["人名"])
            chapter_path = root / "第一章.txt"
            chapter_path.write_text("正文", encoding="utf-8")
            chapter = store.import_chapters(project["id"], [chapter_path]).imported[0]
            dialog = ResidualReviewDialog(store, project, chapter, ["短词"])
            self.assertEqual("ignore", dialog.edits()[0]["action"])

    def test_residual_review_dialog_apply_accepts_valid_manual_name(self):
        """Catches ApplyRole clicks that do not submit the review dialog."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project("项目", "日语", "日语", ["人名"])
            chapter_path = root / "第一章.txt"
            chapter_path.write_text("正文", encoding="utf-8")
            chapter = store.import_chapters(project["id"], [chapter_path]).imported[0]
            dialog = ResidualReviewDialog(store, project, chapter, ["莉娜"])
            row = dialog._rows[0]
            row["action"].setCurrentIndex(1)
            row["target"].setText("美玲")

            dialog.findChild(QDialogButtonBox).button(
                QDialogButtonBox.StandardButton.Apply
            ).click()

            self.assertEqual(QDialog.DialogCode.Accepted, dialog.result())

    def test_review_edits_only_queue_explicit_api_retry_sources(self):
        local, retry_sources = split_review_edits([
            {"action": "ignore", "source": "短词"},
            {"action": "map", "source": "怪物", "target": "冥界骑士"},
            {"action": "retry", "source": "漏改名"},
        ])
        self.assertEqual(["短词", "怪物"], [item["source"] for item in local])
        self.assertEqual(["漏改名"], retry_sources)

    def test_import_selects_mode_once_and_translation_displays_it_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "项目", "日语", "日语", ["人名"],
                processing_mode="日语原文 → 改名润色",
            )
            import_tab = ProjectImportTab(store)
            import_tab.existing_radio.setChecked(True)
            import_tab.refresh_projects(project["id"])
            translation_tab = TranslationTab(store, APIConfigStore(root / "api.json"))

            self.assertFalse(import_tab.processing_mode_combo.isEnabled())
            self.assertEqual(
                "日语原文 → 改名润色",
                import_tab.processing_mode_combo.currentText(),
            )
            self.assertFalse(translation_tab.mode_display.isEnabled())
            self.assertEqual(
                "日语原文 → 改名润色",
                translation_tab.mode_display.text(),
            )

    def test_legacy_project_can_save_its_one_time_mode_without_new_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project("旧项目", "日语", "日语", ["人名"])
            tab = ProjectImportTab(store)
            tab.existing_radio.setChecked(True)
            tab.refresh_projects(project["id"])
            tab.processing_mode_combo.setCurrentText("日语原文 → 改名润色")

            with patch("novel_localizer.desktop_app.QMessageBox.information"), patch(
                "novel_localizer.desktop_app.QMessageBox.critical"
            ) as critical:
                tab.import_files()

            critical.assert_not_called()
            self.assertEqual(
                "日语原文 → 改名润色",
                store.load_project(project["id"])["processing_mode"],
            )

    def test_translation_tab_restores_project_processing_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "已有项目", "日语", "日语", ["人名"]
            )
            store.set_processing_mode(project["id"], "日语原文 → 改名润色")
            store.update_processing_settings(project["id"], {
                "mode": "日语原文 → 改名润色",
                "max_output_tokens": 32000,
                "extra_instructions": "人名必须彻底改写。",
                "save_extra_instructions": True,
            })

            tab = TranslationTab(store, APIConfigStore(root / "api.json"))

            self.assertEqual("日语原文 → 改名润色", tab.mode_display.text())
            self.assertEqual(32000, tab.max_tokens.value())
            self.assertTrue(tab.save_extra_instructions.isChecked())
            self.assertEqual("人名必须彻底改写。", tab.extra_instructions.toPlainText())

    def test_translation_tab_has_queue_management_and_vertical_splitter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tab = TranslationTab(
                ProjectStore(root), APIConfigStore(root / "api.json")
            )
            self.assertEqual("查看替换清单", tab.view_replacements_button.text())
            self.assertEqual("删除所选章节", tab.delete_chapters_button.text())
            self.assertEqual(Qt.Orientation.Vertical, tab.content_splitter.orientation())
            self.assertTrue(tab.content_splitter.childrenCollapsible() is False)

    def test_chapter_queue_supports_extended_row_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            tab = TranslationTab(
                ProjectStore(Path(temp)), APIConfigStore(Path(temp) / "api.json")
            )

            self.assertEqual(
                QAbstractItemView.SelectionMode.ExtendedSelection,
                tab.chapter_table.selectionMode(),
            )

    def test_review_context_menu_is_available_only_for_reviewable_chapter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project("项目", "日语", "日语", ["人名"])
            chapter_path = root / "第一章.txt"
            chapter_path.write_text("正文", encoding="utf-8")
            chapter = store.import_chapters(project["id"], [chapter_path]).imported[0]
            store.confirm_replacement(
                project["id"], "人名", "莉娜", "美玲", allow_user_override=True
            )
            store.save_translation(
                project["id"], chapter["id"], "莉娜来了。",
                profile_id="", provider="", model="", usage={}, replacements_added=0,
                warnings=[], mode="", review_status="需要复核",
                residual_sources=["莉娜"], memory_version=0, skill_hash="", skill_files=[],
            )
            tab = TranslationTab(store, APIConfigStore(root / "api.json"))

            review_menu = tab.chapter_context_menu_for_row(0)
            self.assertEqual(["复核残留原词…"], [action.text() for action in review_menu.actions()])

            store.save_translation(
                project["id"], chapter["id"], "美玲来了。",
                profile_id="", provider="", model="", usage={}, replacements_added=0,
                warnings=[], mode="", review_status="完成",
                residual_sources=[], memory_version=0, skill_hash="", skill_files=[],
            )
            tab.refresh_data()
            self.assertEqual([], tab.chapter_context_menu_for_row(0).actions())

    def test_toggle_all_checked_selects_all_then_clears_all(self):
        with tempfile.TemporaryDirectory() as temp:
            tab = TranslationTab(
                ProjectStore(Path(temp)), APIConfigStore(Path(temp) / "api.json")
            )
            tab.chapter_table.setRowCount(2)
            for row in range(2):
                item = QTableWidgetItem()
                item.setCheckState(Qt.CheckState.Unchecked)
                tab.chapter_table.setItem(row, 0, item)

            tab.toggle_all_checked()
            self.assertEqual(2, len(tab.selected_chapter_ids()))
            tab.toggle_all_checked()
            self.assertEqual([], tab.selected_chapter_ids())

    def test_toggle_all_checked_selects_all_when_queue_is_partially_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            tab = TranslationTab(
                ProjectStore(Path(temp)), APIConfigStore(Path(temp) / "api.json")
            )
            tab.chapter_table.setRowCount(2)
            for row, state in enumerate(
                (Qt.CheckState.Checked, Qt.CheckState.Unchecked)
            ):
                item = QTableWidgetItem()
                item.setCheckState(state)
                tab.chapter_table.setItem(row, 0, item)

            tab.toggle_all_checked()
            self.assertEqual(2, len(tab.selected_chapter_ids()))

    def test_existing_project_loads_saved_skills(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ProjectStore(Path(temp))
            project = store.create_project(
                "已有项目", "中文（简体）", "日语", ["人名"]
            )
            created = store.add_skill_text(project["id"], "旧规则。")
            tab = ProjectImportTab(store)

            tab.existing_radio.setChecked(True)
            tab.refresh_projects(project["id"])

            self.assertEqual(1, tab.existing_skills.count())
            self.assertEqual(
                created[0]["id"],
                tab.existing_skills.item(0).data(Qt.ItemDataRole.UserRole),
            )

    def test_existing_project_can_save_only_a_new_skill_without_txt(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ProjectStore(Path(temp))
            project = store.create_project(
                "已有项目", "中文（简体）", "日语", ["人名"]
            )
            store.set_processing_mode(project["id"], "中文小说 → 日语本地化")
            tab = ProjectImportTab(store)
            tab.existing_radio.setChecked(True)
            tab.refresh_projects(project["id"])
            tab.skill_files.text_edit.setPlainText("新规则。")

            with patch("novel_localizer.desktop_app.QMessageBox.information"), patch(
                "novel_localizer.desktop_app.QMessageBox.critical"
            ) as critical:
                tab.import_files()

            self.assertEqual("新规则。", store.read_project_skills(project["id"])[0]["content"])
            critical.assert_not_called()

    def test_existing_skill_can_be_edited_from_management_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ProjectStore(Path(temp))
            project = store.create_project(
                "已有项目", "中文（简体）", "日语", ["人名"]
            )
            store.add_skill_text(project["id"], "旧规则。")
            tab = ProjectImportTab(store)
            tab.existing_radio.setChecked(True)
            tab.refresh_projects(project["id"])
            tab.existing_skills.item(0).setSelected(True)
            tab.existing_skill_editor.setPlainText("新规则。")

            with patch("novel_localizer.desktop_app.QMessageBox.information"):
                tab.save_selected_skill()

            self.assertEqual("新规则。", store.read_project_skills(project["id"])[0]["content"])

    def test_multiple_existing_skills_can_be_deleted_from_management_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ProjectStore(Path(temp))
            project = store.create_project(
                "已有项目", "中文（简体）", "日语", ["人名"]
            )
            store.add_skill_text(project["id"], "规则一。")
            store.add_skill_text(project["id"], "规则二。")
            tab = ProjectImportTab(store)
            tab.existing_radio.setChecked(True)
            tab.refresh_projects(project["id"])
            tab.existing_skills.selectAll()

            with patch(
                "novel_localizer.desktop_app.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ), patch("novel_localizer.desktop_app.QMessageBox.information"):
                tab.delete_selected_skills()

            self.assertEqual([], store.read_project_skills(project["id"]))


if __name__ == "__main__":
    unittest.main()
