from __future__ import annotations

import unittest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QAbstractItemView,
    QTableWidgetItem,
)

from novel_localizer.desktop_app import (
    FileListBox, MainWindow, ProjectImportTab, ResidualReviewDialog, TranslationTab,
    TranslationWorker, _acquire_instance_lock, available_name_candidates,
    chapters_requiring_review, format_automatic_review_report,
    format_automatic_review_summary, split_review_edits,
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

    def test_only_one_instance_lock_can_be_held_for_the_same_data_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = _acquire_instance_lock(root)
            second = _acquire_instance_lock(root)

            self.assertIsNotNone(first)
            self.assertIsNone(second)

            first.unlock()
            third = _acquire_instance_lock(root)
            self.assertIsNotNone(third)
            third.unlock()

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

    def test_translation_tab_has_no_manual_residual_review_button(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tab = TranslationTab(ProjectStore(root), APIConfigStore(root / "api.json"))

            self.assertFalse(hasattr(tab, "polish_reviews_button"))

    def test_cancel_translation_immediately_unlocks_api_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            tab = TranslationTab(
                ProjectStore(Path(temp)), APIConfigStore(Path(temp) / "api.json")
            )
            cancelled = []
            tab.worker = type(
                "Worker", (), {"cancel": lambda self: cancelled.append(True)}
            )()
            tab.profile_combo.setEnabled(False)

            tab.cancel_translation()

            self.assertEqual([True], cancelled)
            self.assertTrue(tab.profile_combo.isEnabled())

    def test_cancelled_worker_restores_interrupted_chapter_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "项目", "中文", "日语", ["人名"], processing_mode="润色"
            )
            source = root / "第一章.txt"
            source.write_text("正文", encoding="utf-8")
            chapter = store.import_chapters(project["id"], [source]).imported[0]
            store.mark_chapters_queued(project["id"], [chapter["id"]])
            worker = TranslationWorker(
                store,
                project["id"],
                [chapter["id"]],
                {
                    "id": "p", "provider": "OpenAI", "base_url": "https://example.test",
                    "model": "m", "api_key": "key", "timeout_seconds": 1,
                },
                "润色", "", 100,
            )

            worker.cancel()
            worker.run()

            self.assertEqual(
                "待处理", store.get_chapter(project["id"], chapter["id"])["status"]
            )

    def test_worker_finishes_when_failure_state_cannot_be_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "项目", "中文", "日语", ["人名"], processing_mode="润色"
            )
            source = root / "第一章.txt"
            source.write_text("正文", encoding="utf-8")
            chapter = store.import_chapters(project["id"], [source]).imported[0]
            completed = []
            logs = []
            worker = TranslationWorker(
                store,
                project["id"],
                [chapter["id"]],
                {
                    "id": "p", "provider": "OpenAI", "base_url": "https://example.test",
                    "model": "m", "api_key": "key", "timeout_seconds": 1,
                },
                "润色", "", 100,
            )
            worker.task_finished.connect(completed.append)
            worker.log_message.connect(logs.append)

            with patch(
                "novel_localizer.desktop_app.TranslationEngine.translate_chapter",
                side_effect=RuntimeError("translation failed"),
            ), patch.object(
                store,
                "record_translation_failure",
                side_effect=PermissionError("manifest locked"),
            ):
                worker.run()

            self.assertEqual([False], completed)
            self.assertTrue(any("无法记录失败状态" in message for message in logs))

    def test_worker_does_not_replay_a_saved_translation_for_residual_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project("项目", "中文", "日语", ["人名"], processing_mode="翻译")
            chapter = store.import_chapter_text(project["id"], "正文").imported[0]
            profile = {"id": "p", "provider": "OpenAI", "model": "m", "api_key": "key"}
            worker = TranslationWorker(store, project["id"], [chapter["id"]], profile, "翻译", "", 100)

            def save_result(residuals):
                record = store.save_translation(
                    project["id"], chapter["id"], "译文", profile_id="p", provider="OpenAI",
                    model="m", review_status="需要复核" if residuals else "完成",
                    residual_sources=residuals,
                )
                return {"translated_text": "译文", "new_replacements": 0, "warnings": [],
                        "usage": {}, "record": record, "residual_sources": residuals,
                        "review_status": "需要复核" if residuals else "完成"}

            with patch("novel_localizer.desktop_app.TranslationEngine.translate_chapter", side_effect=lambda *args, **kwargs: save_result(["莉娜"])), patch(
                "novel_localizer.desktop_app.TranslationEngine.polish_review_chapter",
            ) as polish:
                worker.run()

            self.assertEqual(0, polish.call_count)

    def test_worker_keeps_chapters_even_if_a_legacy_result_has_residual_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project("项目", "中文", "日语", ["人名"], processing_mode="翻译")
            chapters = [store.import_chapter_text(project["id"], f"正文{i}").imported[0] for i in range(20)]
            profile = {"id": "p", "provider": "OpenAI", "model": "m", "api_key": "key"}
            worker = TranslationWorker(store, project["id"], [item["id"] for item in chapters], profile, "翻译", "", 100)

            def translate(project_id, chapter_id, *args, **kwargs):
                residuals = ["莉娜"] if chapter_id == chapters[0]["id"] else []
                record = store.save_translation(project_id, chapter_id, "译文", profile_id="p", provider="OpenAI", model="m", review_status="需要复核" if residuals else "完成", residual_sources=residuals)
                return {"translated_text": "译文", "new_replacements": 0, "warnings": [], "usage": {}, "record": record, "residual_sources": residuals, "review_status": "需要复核" if residuals else "完成"}

            with patch("novel_localizer.desktop_app.TranslationEngine.translate_chapter", side_effect=translate), patch(
                "novel_localizer.desktop_app.TranslationEngine.polish_review_chapter"
            ) as polish:
                worker.run()

            self.assertEqual(0, polish.call_count)
            self.assertIsNotNone(store.get_chapter(project["id"], chapters[0]["id"]))
            self.assertEqual(20, len(store.load_project(project["id"])["chapters"]))

    def test_worker_does_not_mark_a_saved_result_as_failed_for_residual_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project("项目", "中文", "日语", ["人名"], processing_mode="翻译")
            chapters = [store.import_chapter_text(project["id"], f"正文{i}").imported[0] for i in range(2)]
            profile = {"id": "p", "provider": "OpenAI", "model": "m", "api_key": "key"}
            worker = TranslationWorker(store, project["id"], [item["id"] for item in chapters], profile, "翻译", "", 100)

            def translate(project_id, chapter_id, *args, **kwargs):
                residuals = ["莉娜"] if chapter_id == chapters[0]["id"] else []
                record = store.save_translation(project_id, chapter_id, "最后译文", profile_id="p", provider="OpenAI", model="m", review_status="需要复核" if residuals else "完成", residual_sources=residuals)
                return {"translated_text": "最后译文", "new_replacements": 0, "warnings": [], "usage": {}, "record": record, "residual_sources": residuals, "review_status": "需要复核" if residuals else "完成"}

            with patch("novel_localizer.desktop_app.TranslationEngine.translate_chapter", side_effect=translate), patch(
                "novel_localizer.desktop_app.TranslationEngine.polish_review_chapter"
            ) as polish:
                worker.run()

            saved = store.get_chapter(project["id"], chapters[0]["id"])
            self.assertEqual(0, polish.call_count)
            self.assertEqual("需要复核", saved["status"])
            self.assertEqual("", saved["last_error"])
            self.assertEqual("最后译文\n", store.read_latest_translation(project["id"], chapters[0]["id"]))

    def test_worker_cancellation_after_residual_retry_never_deletes_chapter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project("项目", "中文", "日语", ["人名"], processing_mode="翻译")
            chapter = store.import_chapter_text(project["id"], "正文").imported[0]
            profile = {"id": "p", "provider": "OpenAI", "model": "m", "api_key": "key"}
            worker = TranslationWorker(store, project["id"], [chapter["id"]], profile, "翻译", "", 100)
            def save(residuals):
                record = store.save_translation(project["id"], chapter["id"], "译文", profile_id="p", provider="OpenAI", model="m", review_status="需要复核", residual_sources=residuals)
                return {"translated_text": "译文", "new_replacements": 0, "warnings": [], "usage": {}, "record": record, "residual_sources": residuals, "review_status": "需要复核"}
            def polish(*args, **kwargs):
                worker.cancel()
                return save(["莉娜"])
            with patch("novel_localizer.desktop_app.TranslationEngine.translate_chapter", side_effect=lambda *args, **kwargs: save(["莉娜"])), patch("novel_localizer.desktop_app.TranslationEngine.polish_review_chapter", side_effect=polish):
                worker.run()
            self.assertIsNotNone(store.get_chapter(project["id"], chapter["id"]))

    def test_closing_window_hides_to_tray_while_translation_is_running(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "novel_localizer.desktop_app.application_data_root", return_value=Path(temp)
        ):
            window = MainWindow()
            window.translation_tab.worker = type(
                "RunningWorker", (), {"isRunning": lambda self: True}
            )()
            event = QCloseEvent()

            window.closeEvent(event)

            self.assertFalse(event.isAccepted())
            self.assertTrue(window.isHidden())
            self.assertTrue(window.tray_icon.isVisible())
            window.tray_icon.hide()

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

    def test_review_edits_queue_api_retry_without_a_manual_mapping(self):
        local, retry_sources = split_review_edits([
            {"action": "ignore", "source": "短词"},
            {"action": "map", "source": "怪物", "target": "冥界骑士"},
            {"action": "retry", "source": "漏改名"},
        ])
        self.assertEqual(["短词", "怪物"], [item["source"] for item in local])
        self.assertEqual(["漏改名"], retry_sources)

    def test_automatic_review_summary_reports_final_chapter_states(self):
        summary = format_automatic_review_summary({
            "chapters": [
                {"status": "已翻译"}, {"status": "已翻译"},
                {"status": "需要复核"}, {"status": "翻译失败"},
            ],
        }, cancelled=False)

        self.assertEqual(
            "全书自动复核结束：已翻译 2 章；需要复核 1 章；翻译失败 1 章。",
            summary,
        )

    def test_automatic_review_retranslates_only_chapters_that_need_review(self):
        chapter_ids = chapters_requiring_review({
            "chapters": [
                {"id": "done", "translations": [{"review_status": "完成"}]},
                {"id": "review", "translations": [{"review_status": "需要复核"}]},
                {"id": "missing", "translations": []},
            ],
        })

        self.assertEqual(["review"], chapter_ids)

    def test_automatic_review_report_distinguishes_success_from_api_failure(self):
        summary = "全书自动复核结束：已翻译 2 章；需要复核 1 章；翻译失败 0 章。"
        success = format_automatic_review_report({
            "success": True,
            "mappings": [{"source": "司", "target": "美咲", "category": "人名"}],
        }, summary)
        failure = format_automatic_review_report({
            "success": False, "error": "HTTP 403：额度不足",
        }, summary)

        self.assertIn("API 实体复核完成", success)
        self.assertIn("司 → 美咲", success)
        self.assertIn("API 实体复核失败", failure)
        self.assertIn("HTTP 403", failure)

    def test_completed_chapter_does_not_queue_manual_residual_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project("项目", "日语", "日语", ["人名"])
            chapter_path = root / "第一章.txt"
            chapter_path.write_text("正文", encoding="utf-8")
            chapter = store.import_chapters(project["id"], [chapter_path]).imported[0]
            tab = TranslationTab(store, APIConfigStore(root / "api.json"))
            tab.chapter_succeeded(chapter["id"], {
                "translated_text": "译文", "warnings": [], "residual_sources": [],
            })

            self.assertEqual("已翻译", tab.chapter_table.item(0, 3).text())
            self.assertFalse(hasattr(tab, "pending_retries"))

    def test_residual_review_api_retry_does_not_require_a_name_or_description(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project("项目", "日语", "日语", ["人名"])
            chapter_path = root / "第一章.txt"
            chapter_path.write_text("正文", encoding="utf-8")
            chapter = store.import_chapters(project["id"], [chapter_path]).imported[0]
            dialog = ResidualReviewDialog(store, project, chapter, ["司"])
            row = dialog._rows[0]
            row["action"].setCurrentIndex(2)

            self.assertFalse(row["target"].isEnabled())
            self.assertFalse(row["notes"].isEnabled())
            dialog.findChild(QDialogButtonBox).button(
                QDialogButtonBox.StandardButton.Apply
            ).click()

            self.assertEqual(QDialog.DialogCode.Accepted, dialog.result())

    def test_import_selects_mode_once_and_translation_displays_it_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "项目", "日语", "日语", ["人名"],
                processing_mode="润色",
            )
            import_tab = ProjectImportTab(store)
            import_tab.existing_radio.setChecked(True)
            import_tab.refresh_projects(project["id"])
            translation_tab = TranslationTab(store, APIConfigStore(root / "api.json"))

            self.assertFalse(import_tab.processing_mode_combo.isEnabled())
            self.assertEqual(
                "润色",
                import_tab.processing_mode_combo.currentText(),
            )
            self.assertFalse(translation_tab.mode_display.isEnabled())
            self.assertEqual(
                "润色",
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
            tab.processing_mode_combo.setCurrentText("润色")

            with patch("novel_localizer.desktop_app.QMessageBox.information"), patch(
                "novel_localizer.desktop_app.QMessageBox.critical"
            ) as critical:
                tab.import_files()

            critical.assert_not_called()
            self.assertEqual(
                "润色",
                store.load_project(project["id"])["processing_mode"],
            )

    def test_import_tab_saves_project_rename_setting_without_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "已有项目", "中文（简体）", "中文（繁体）", ["人名"],
                processing_mode="润色", proper_noun_localization=False,
            )
            tab = ProjectImportTab(store)
            tab.existing_radio.setChecked(True)
            tab.refresh_projects(project["id"])

            self.assertFalse(tab.rename_mode_combo.currentData())
            tab.rename_mode_combo.setCurrentIndex(
                tab.rename_mode_combo.findData(True)
            )

            with patch(
                "novel_localizer.desktop_app.QMessageBox.information"
            ) as information, patch(
                "novel_localizer.desktop_app.QMessageBox.critical"
            ) as critical:
                tab.save_project_settings()

            critical.assert_not_called()
            self.assertTrue(
                store.load_project(project["id"])["proper_noun_localization"]
            )
            self.assertIn("已有项目", information.call_args.args[2])

    def test_import_tab_creates_project_with_rename_setting_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            tab = ProjectImportTab(store)
            tab.name_edit.setText("关闭改名的新项目")
            tab.processing_mode_combo.setCurrentText("润色")
            tab.rename_mode_combo.setCurrentIndex(
                tab.rename_mode_combo.findData(False)
            )
            tab.chapter_files.text_edit.setPlainText("曹操来到颍川。")

            with patch(
                "novel_localizer.desktop_app.QMessageBox.information"
            ), patch(
                "novel_localizer.desktop_app.QMessageBox.critical"
            ) as critical:
                tab.import_files()

            critical.assert_not_called()
            saved = store.list_projects()[0]
            self.assertFalse(saved["proper_noun_localization"])

    def test_switching_from_existing_project_resets_new_project_rename_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "关闭改名的旧项目", "中文", "日语", ["人名"],
                processing_mode="翻译", proper_noun_localization=False,
            )
            tab = ProjectImportTab(store)
            tab.existing_radio.setChecked(True)
            tab.refresh_projects(project["id"])
            self.assertFalse(tab.rename_mode_combo.currentData())

            tab.new_radio.setChecked(True)

            self.assertTrue(tab.rename_mode_combo.currentData())

    def test_translation_tab_restores_project_processing_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "已有项目", "日语", "日语", ["人名"]
            )
            store.set_processing_mode(project["id"], "润色")
            store.update_processing_settings(project["id"], {
                "mode": "润色",
                "max_output_tokens": 32000,
                "extra_instructions": "人名必须彻底改写。",
                "save_extra_instructions": True,
            })

            tab = TranslationTab(store, APIConfigStore(root / "api.json"))

            self.assertEqual("润色", tab.mode_display.text())
            self.assertEqual(32000, tab.max_tokens.value())
            self.assertTrue(tab.save_extra_instructions.isChecked())
            self.assertEqual("人名必须彻底改写。", tab.extra_instructions.toPlainText())

    def test_processing_controls_save_only_on_button_click(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "显式保存项目", "中文", "日语", ["人名"],
                processing_mode="翻译",
            )
            secret_store = type(
                "SecretStore",
                (),
                {
                    "protect": staticmethod(lambda value: f"encrypted:{value}"),
                    "unprotect": staticmethod(lambda _value: "test-key"),
                },
            )()
            api_store = APIConfigStore(root, secret_store=secret_store)
            first = api_store.upsert_profile(
                profile_id=None, name="配置一", provider="OpenAI",
                base_url="https://one.test/v1", model="model-one",
                api_key="key-one", timeout_seconds=30,
            )
            second = api_store.upsert_profile(
                profile_id=None, name="配置二", provider="OpenAI",
                base_url="https://two.test/v1", model="model-two",
                api_key="key-two", timeout_seconds=30,
            )
            store.update_processing_settings(project["id"], {
                "profile_id": first["id"], "max_output_tokens": 24000,
            })
            tab = TranslationTab(store, api_store)

            tab.profile_combo.setCurrentIndex(
                tab.profile_combo.findData(second["id"])
            )
            tab.max_tokens.setValue(32000)

            before_save = store.load_project(project["id"])["processing_settings"]
            self.assertEqual(first["id"], before_save["profile_id"])
            self.assertEqual(24000, before_save["max_output_tokens"])
            self.assertEqual("同时保存额外要求", tab.save_extra_instructions.text())

            with patch(
                "novel_localizer.desktop_app.QMessageBox.information"
            ) as information:
                tab.save_settings_button.click()

            after_save = store.load_project(project["id"])["processing_settings"]
            self.assertEqual(second["id"], after_save["profile_id"])
            self.assertEqual(32000, after_save["max_output_tokens"])
            self.assertIn("显式保存项目", information.call_args.args[2])

    def test_projects_restore_distinct_processing_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            first_project = store.create_project(
                "项目甲", "中文", "日语", ["人名"], processing_mode="翻译"
            )
            second_project = store.create_project(
                "项目乙", "中文", "日语", ["人名"], processing_mode="翻译"
            )
            missing_project = store.create_project(
                "配置已删除", "中文", "日语", ["人名"], processing_mode="翻译"
            )
            secret_store = type(
                "SecretStore",
                (),
                {
                    "protect": staticmethod(lambda value: f"encrypted:{value}"),
                    "unprotect": staticmethod(lambda _value: "test-key"),
                },
            )()
            api_store = APIConfigStore(root, secret_store=secret_store)
            first_profile = api_store.upsert_profile(
                profile_id=None, name="甲配置", provider="OpenAI",
                base_url="https://one.test/v1", model="model-one",
                api_key="key-one", timeout_seconds=30,
            )
            second_profile = api_store.upsert_profile(
                profile_id=None, name="乙配置", provider="OpenAI",
                base_url="https://two.test/v1", model="model-two",
                api_key="key-two", timeout_seconds=30,
            )
            store.update_processing_settings(first_project["id"], {
                "profile_id": first_profile["id"], "max_output_tokens": 12000,
            })
            store.update_processing_settings(second_project["id"], {
                "profile_id": second_profile["id"], "max_output_tokens": 36000,
            })
            store.update_processing_settings(missing_project["id"], {
                "profile_id": "deleted-profile", "max_output_tokens": 48000,
            })
            tab = TranslationTab(store, api_store)

            tab.project_combo.setCurrentIndex(
                tab.project_combo.findData(first_project["id"])
            )
            self.assertEqual(first_profile["id"], tab.profile_combo.currentData())
            self.assertEqual(12000, tab.max_tokens.value())

            tab.project_combo.setCurrentIndex(
                tab.project_combo.findData(second_project["id"])
            )
            self.assertEqual(second_profile["id"], tab.profile_combo.currentData())
            self.assertEqual(36000, tab.max_tokens.value())

            tab.project_combo.setCurrentIndex(
                tab.project_combo.findData(missing_project["id"])
            )
            self.assertIsNone(tab.profile_combo.currentData())
            self.assertEqual(48000, tab.max_tokens.value())

    def test_invalid_processing_settings_restore_safe_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "损坏配置", "中文", "日语", ["人名"], processing_mode="翻译"
            )
            store.update_processing_settings(project["id"], {
                "max_output_tokens": True,
                "save_extra_instructions": "false",
                "extra_instructions": "不应恢复",
            })

            tab = TranslationTab(store, APIConfigStore(root / "api.json"))

            self.assertEqual(24000, tab.max_tokens.value())
            self.assertFalse(tab.save_extra_instructions.isChecked())
            self.assertEqual("", tab.extra_instructions.toPlainText())

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

    def test_translation_controls_and_preview_can_be_collapsed(self):
        with tempfile.TemporaryDirectory() as temp:
            tab = TranslationTab(
                ProjectStore(Path(temp)), APIConfigStore(Path(temp) / "api.json")
            )

            self.assertFalse(tab.processing_controls_body.isHidden())
            self.assertEqual("", tab.processing_controls_toggle.text())
            self.assertEqual(Qt.ArrowType.UpArrow, tab.processing_controls_toggle.arrowType())
            tab.toggle_processing_controls()
            self.assertTrue(tab.processing_controls_body.isHidden())
            self.assertEqual(Qt.ArrowType.DownArrow, tab.processing_controls_toggle.arrowType())

            self.assertFalse(tab.preview_body.isHidden())
            self.assertEqual("", tab.preview_toggle.text())
            self.assertEqual(Qt.ArrowType.UpArrow, tab.preview_toggle.arrowType())
            tab.toggle_preview()
            self.assertTrue(tab.preview_body.isHidden())
            self.assertEqual(Qt.ArrowType.LeftArrow, tab.preview_toggle.arrowType())
            self.assertTrue(tab.preview_title.isHidden())
            self.assertEqual(52, tab.preview_card.maximumWidth())

    def test_start_translation_marks_selected_chapters_as_queued(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            project = store.create_project(
                "项目", "日语", "日语", ["人名"],
                processing_mode="润色",
            )
            chapter_path = root / "第一章.txt"
            chapter_path.write_text("正文", encoding="utf-8")
            chapter = store.import_chapters(project["id"], [chapter_path]).imported[0]

            store.mark_chapters_queued(project["id"], [chapter["id"]])

            self.assertEqual(
                "队列中", store.get_chapter(project["id"], chapter["id"])["status"]
            )

            store.mark_chapter_processing(project["id"], chapter["id"])

            self.assertEqual(
                "处理中", store.get_chapter(project["id"], chapter["id"])["status"]
            )

    def test_start_translation_opens_confirmation_when_skill_analysis_is_pending(self):
        """A pending Skill must not prevent the start confirmation from opening."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ProjectStore(root)
            api_store = APIConfigStore(
                root,
                secret_store=type(
                    "SecretStore",
                    (),
                    {
                        "protect": staticmethod(lambda value: f"encrypted:{value}"),
                        "unprotect": staticmethod(lambda _value: "test-key"),
                    },
                )(),
            )
            project = store.create_project(
                "项目", "日语", "日语", ["人名"], processing_mode="润色"
            )
            chapter_path = root / "第一章.txt"
            chapter_path.write_text("正文", encoding="utf-8")
            store.import_chapters(project["id"], [chapter_path])
            store.add_skill_text(project["id"], "使用敬语。")
            profile = api_store.upsert_profile(
                profile_id=None,
                name="测试配置",
                provider="OpenAI 兼容接口",
                base_url="https://example.test/v1",
                model="test-model",
                api_key="test-key",
                timeout_seconds=30,
            )
            tab = TranslationTab(store, api_store)
            tab.profile_combo.setCurrentIndex(tab.profile_combo.findData(profile["id"]))

            with patch(
                "novel_localizer.desktop_app.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ) as question:
                tab.start_translation()

            self.assertEqual(1, question.call_count)
            self.assertIn("1 份新增或已修改的 Skill", question.call_args.args[2])

    def test_chapter_queue_supports_extended_row_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            tab = TranslationTab(
                ProjectStore(Path(temp)), APIConfigStore(Path(temp) / "api.json")
            )

            self.assertEqual(
                QAbstractItemView.SelectionMode.ExtendedSelection,
                tab.chapter_table.selectionMode(),
            )

    def test_review_context_menu_has_no_manual_residual_action(self):
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
            self.assertEqual([], [action.text() for action in review_menu.actions()])

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
            store.set_processing_mode(project["id"], "翻译")
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
