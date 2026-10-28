from __future__ import annotations

import os
import sys
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QLockFile, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QSystemTrayIcon,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QAbstractItemView,
    QHeaderView,
)

from . import __version__
from .domain import (
    DEFAULT_CATEGORIES,
    LANGUAGE_OPTIONS,
    PROVIDER_DEFAULTS,
    TRANSLATION_MODES,
    ValidationError,
)
from .storage import APIConfigStore, ProjectStore
from .translator import (
    ModelClient,
    TranslationCancelled,
    TranslationEngine,
    format_batch_start_log,
    format_usage_summary,
)


APP_STYLE = """
QWidget {
    color: #182033;
    font-family: "Microsoft YaHei UI";
    font-size: 13px;
}
QMainWindow, QWidget#AppRoot, QScrollArea, QScrollArea > QWidget > QWidget {
    background: #F3F5F9;
}
QFrame#Header {
    background: #FFFFFF;
    border-bottom: 1px solid #E4E8F0;
}
QLabel#BrandMark {
    color: white;
    background: #5B5BD6;
    border-radius: 11px;
    font-family: "KaiTi";
    font-size: 20px;
    font-weight: 700;
}
QLabel#BrandTitle { font-size: 19px; font-weight: 700; }
QLabel#BrandSub, QLabel#Muted, QLabel#Hint { color: #677085; }
QLabel#BrandSub, QLabel#Hint { font-size: 11px; }
QLabel#Version {
    color: #4646B8;
    background: #EEEEFF;
    border: 1px solid #DCDCFF;
    border-radius: 12px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 700;
}
QFrame#Card, QGroupBox {
    background: #FFFFFF;
    border: 1px solid #E4E8F0;
    border-radius: 12px;
}
QGroupBox {
    margin-top: 10px;
    padding-top: 17px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 13px;
    padding: 0 6px;
    color: #414A5F;
}
QLabel#SectionTitle { font-size: 18px; font-weight: 700; }
QLabel#CardTitle { font-size: 15px; font-weight: 700; }
QLabel#Eyebrow {
    color: #5B5BD6;
    font-size: 10px;
    font-weight: 800;
}
QLineEdit, QComboBox, QSpinBox, QTextEdit, QListWidget, QTableWidget {
    background: #FFFFFF;
    border: 1px solid #CBD2DF;
    border-radius: 7px;
    padding: 7px;
    selection-background-color: #E7E7FF;
    selection-color: #3434A2;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus,
QListWidget:focus, QTableWidget:focus {
    border: 1px solid #5B5BD6;
}
QLineEdit:disabled, QComboBox:disabled {
    color: #8D95A5;
    background: #F2F4F7;
}
QComboBox, QSpinBox { min-height: 24px; }
QPushButton {
    min-height: 35px;
    padding: 0 14px;
    background: #FFFFFF;
    border: 1px solid #CBD2DF;
    border-radius: 7px;
    font-weight: 600;
}
QPushButton:hover { background: #F7F8FB; border-color: #AEB6C7; }
QPushButton:pressed { background: #ECEEF3; }
QToolButton {
    background: #FFFFFF;
    border: 1px solid #CBD2DF;
    border-radius: 7px;
}
QToolButton:hover { background: #F7F8FB; border-color: #AEB6C7; }
QToolButton:pressed { background: #ECEEF3; }
QPushButton#Primary {
    color: white;
    background: #5B5BD6;
    border: 1px solid #5B5BD6;
    font-weight: 700;
}
QPushButton#Primary:hover { background: #4848BE; }
QPushButton#Danger {
    color: #B42335;
    background: #FFF2F3;
    border-color: #F3C6CB;
}
QPushButton:disabled { color: #98A0AF; background: #ECEEF2; border-color: #DDE1E8; }
QRadioButton, QCheckBox { spacing: 7px; }
QTabWidget::pane { border: 0; background: #F3F5F9; }
QTabBar::tab {
    min-width: 130px;
    min-height: 42px;
    padding: 0 16px;
    color: #687084;
    background: #FFFFFF;
    border: 0;
    border-bottom: 3px solid transparent;
    font-weight: 600;
}
QTabBar::tab:selected {
    color: #4646B8;
    background: #F8F8FF;
    border-bottom: 3px solid #5B5BD6;
    font-weight: 700;
}
QHeaderView::section {
    color: #4C5569;
    background: #F7F8FB;
    border: 0;
    border-bottom: 1px solid #E4E8F0;
    padding: 8px;
    font-weight: 700;
}
QTableWidget { gridline-color: #EDF0F5; padding: 0; }
QProgressBar {
    min-height: 14px;
    text-align: center;
    color: #4B5262;
    background: #E9ECF2;
    border: 0;
    border-radius: 7px;
}
QProgressBar::chunk { background: #5B5BD6; border-radius: 7px; }
QFrame#Callout {
    background: #EEEEFF;
    border: 1px solid #D8D8FF;
    border-radius: 10px;
}
QFrame#SuccessCallout {
    background: #ECFDF3;
    border: 1px solid #ABEFC6;
    border-radius: 10px;
}
QStatusBar { background: #FFFFFF; color: #687084; }
"""


SAVED_KEY_MASK = "••••••••••••••••"


def application_data_root() -> Path:
    override = os.environ.get("NOVEL_LOCALIZER_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data"


def split_categories(value: str) -> list[str]:
    normalized = value.replace("，", ",").replace("、", ",").replace("\n", ",")
    return list(
        dict.fromkeys(item.strip() for item in normalized.split(",") if item.strip())
    )


def format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value / 1024 / 1024:.1f} MB"


def card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("Card")
    return frame


def label(text: str, object_name: str | None = None) -> QLabel:
    widget = QLabel(text)
    if object_name:
        widget.setObjectName(object_name)
    return widget


class MaskedKeyLineEdit(QLineEdit):
    """已保存的 Key 显示为遮蔽字符；用户输入时自动清空遮蔽内容。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._showing_saved_mask = False
        self.setEchoMode(QLineEdit.EchoMode.Password)

    def show_saved_mask(self):
        self._showing_saved_mask = True
        self.setText(SAVED_KEY_MASK)
        self.setCursorPosition(0)

    def clear_for_new_key(self):
        self._showing_saved_mask = False
        self.clear()

    def has_saved_mask(self) -> bool:
        return self._showing_saved_mask

    def keyPressEvent(self, event):
        if self._showing_saved_mask:
            navigation_keys = {
                Qt.Key.Key_Left,
                Qt.Key.Key_Right,
                Qt.Key.Key_Up,
                Qt.Key.Key_Down,
                Qt.Key.Key_Home,
                Qt.Key.Key_End,
                Qt.Key.Key_Tab,
            }
            is_copy = (
                event.modifiers() & Qt.KeyboardModifier.ControlModifier
                and event.key() == Qt.Key.Key_C
            )
            if event.key() not in navigation_keys and not is_copy:
                self.clear_for_new_key()
        super().keyPressEvent(event)


class ConnectionTestWorker(QThread):
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, profile: dict):
        super().__init__()
        self.profile = profile
        self.cancel_event = threading.Event()

    def run(self):
        try:
            result = ModelClient().test_connection(
                self.profile, cancel_event=self.cancel_event
            )
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class TranslationWorker(QThread):
    progress_changed = Signal(int, int, str)
    chapter_started = Signal(str)
    chapter_succeeded = Signal(str, object)
    chapter_failed = Signal(str, str)
    task_finished = Signal(bool)
    log_message = Signal(str)

    def __init__(
        self,
        store: ProjectStore,
        project_id: str,
        chapter_ids: list[str],
        profile: dict,
        mode: str,
        extra_instructions: str,
        max_output_tokens: int,
    ):
        super().__init__()
        self.store = store
        self.project_id = project_id
        self.chapter_ids = chapter_ids
        self.profile = profile
        self.mode = mode
        self.extra_instructions = extra_instructions
        self.max_output_tokens = max_output_tokens
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def _emit_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_message.emit(f"[{timestamp}] {message}")

    def _replacement_snapshot(self) -> set[tuple[str, str, str]]:
        return {
            (str(category), str(source), str(record.get("target", "")))
            for category, registry in self.store.load_all_replacements(self.project_id).items()
            for source, record in registry.get("entries", {}).items()
        }

    def _resolve_residuals_with_api(
        self, engine: TranslationEngine, chapter_id: str, initial_result: dict
    ) -> tuple[dict, list[str] | None]:
        result = initial_result
        for attempt in range(1, 4):
            if self.cancel_event.is_set():
                raise TranslationCancelled("任务已取消。")
            self._emit_log(f"  发现残留原词，自动 API 润色第 {attempt}/3 次…")
            result = engine.polish_review_chapter(
                self.project_id, chapter_id, self.profile,
                extra_instructions=self.extra_instructions,
                max_output_tokens=self.max_output_tokens,
                cancel_event=self.cancel_event,
            )
            if not result.get("residual_sources"):
                self._emit_log(f"  自动 API 润色第 {attempt}/3 次已解决残留。")
                return result, None
        return result, list(result.get("residual_sources", []))

    def run(self):
        total = len(self.chapter_ids)
        cancelled = False
        cancelled_count = 0
        succeeded = 0
        failed = 0
        batch_started_at = time.monotonic()
        self._emit_log(
            format_batch_start_log(
                self.profile,
                chapter_count=total,
                mode=self.mode,
                max_output_tokens=self.max_output_tokens,
                extra_instructions=self.extra_instructions,
            )
        )

        def record_failure_safely(chapter_id: str, error: Exception | str) -> None:
            error_text = str(error)
            try:
                self.store.record_translation_failure(
                    self.project_id, chapter_id, error_text
                )
            except Exception as persistence_error:
                self._emit_log(
                    f"  无法记录失败状态：{type(persistence_error).__name__}："
                    f"{persistence_error}"
                )
                try:
                    self.store.restore_interrupted_chapters(
                        self.project_id, [chapter_id]
                    )
                except Exception as restore_error:
                    self._emit_log(
                        f"  无法恢复章节状态：{type(restore_error).__name__}："
                        f"{restore_error}"
                    )
            self.chapter_failed.emit(chapter_id, error_text)

        try:
            pending_skills = self.store.pending_skill_analysis(self.project_id)
            if pending_skills:
                self._emit_log(f"正在解析 {len(pending_skills)} 份新增或已修改的 Skill…")
                TranslationEngine(
                    self.store, ModelClient(log_callback=self._emit_log)
                ).analyze_pending_skills(
                    self.project_id, self.profile, self.cancel_event
                )
                self._emit_log("Skill 解析完成；后续未修改时不会再次调用 API。")
        except Exception as exc:
            for chapter_id in self.chapter_ids:
                record_failure_safely(chapter_id, exc)
            self._emit_log(f"Skill 解析失败，未发送正文：{type(exc).__name__}：{exc}")
            self.task_finished.emit(isinstance(exc, TranslationCancelled))
            return
        result_signal_lock = threading.Lock()

        def translate_one(chapter_id):
            if self.cancel_event.is_set():
                raise TranslationCancelled("任务已取消。")
            engine = TranslationEngine(
                self.store,
                ModelClient(log_callback=self._emit_log),
            )
            chapter = self.store.get_chapter(self.project_id, chapter_id)
            name = chapter.get("source_name", "章节")
            self.store.mark_chapter_processing(self.project_id, chapter_id)
            self.chapter_started.emit(chapter_id)
            chapter_started_at = time.monotonic()
            self._emit_log(
                f"开始翻译 {name}；原文字数={chapter.get('characters', '未知')}"
            )
            result = engine.translate_chapter(
                self.project_id,
                chapter_id,
                self.profile,
                mode=self.mode,
                extra_instructions=self.extra_instructions,
                max_output_tokens=self.max_output_tokens,
                cancel_event=self.cancel_event,
                allow_unprepared=False,
            )
            with result_signal_lock:
                self.chapter_succeeded.emit(chapter_id, result)
            elapsed = time.monotonic() - chapter_started_at
            self._emit_log(
                f"完成 {name}：新增 {result['new_replacements']} 条替换记录，"
                f"{len(result['warnings'])} 条提示；"
                f"用量={format_usage_summary(result.get('usage', {}))}；"
                f"耗时={elapsed:.2f}秒"
            )
            return chapter_id, result

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(translate_one, chapter_id): chapter_id
                for chapter_id in self.chapter_ids
            }
            for future in as_completed(futures):
                chapter_id = futures[future]
                try:
                    future.result()
                    succeeded += 1
                except TranslationCancelled:
                    cancelled = True
                    cancelled_count += 1
                    self._emit_log("任务已取消。")
                except Exception as exc:
                    failed += 1
                    record_failure_safely(chapter_id, exc)
                    self._emit_log(
                        f"失败 {chapter_id}：{type(exc).__name__}：{exc}"
                    )
                    self._emit_log("  已跳过失败章节，继续处理队列中的下一章。")
                self.progress_changed.emit(
                    succeeded + failed + cancelled_count, total,
                    f"已处理 {succeeded + failed + cancelled_count}/{total}"
                )
        if cancelled:
            try:
                self.store.restore_interrupted_chapters(
                    self.project_id, self.chapter_ids
                )
            except Exception as exc:
                self._emit_log(f"取消后无法恢复章节状态：{type(exc).__name__}：{exc}")
        batch_elapsed = time.monotonic() - batch_started_at
        self._emit_log(
            f"批次结束：成功={succeeded}，失败={failed}，"
            f"取消={'是' if cancelled else '否'}，总耗时={batch_elapsed:.2f}秒"
        )
        self.task_finished.emit(cancelled)


class FileListBox(QWidget):
    def __init__(
        self,
        title: str,
        hint: str,
        button_text: str,
        file_filter: str,
        parent=None,
    ):
        super().__init__(parent)
        self.file_filter = file_filter
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        title_row = QHBoxLayout()
        title_row.addWidget(label(title, "CardTitle"))
        title_row.addStretch()
        self.count_label = label("0 个", "Hint")
        title_row.addWidget(self.count_label)
        layout.addLayout(title_row)
        layout.addWidget(label(hint, "Hint"))
        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(84)
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setVisible(False)
        layout.addWidget(self.list_widget)
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("可直接在这里粘贴内容……")
        self.text_edit.setMinimumHeight(112)
        layout.addWidget(self.text_edit)
        actions = QHBoxLayout()
        add_button = QPushButton(button_text)
        add_button.clicked.connect(self.choose_files)
        remove_button = QPushButton("移除选中")
        remove_button.clicked.connect(self.remove_selected)
        actions.addWidget(add_button)
        actions.addWidget(remove_button)
        actions.addStretch()
        layout.addLayout(actions)

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择文件", "", self.file_filter
        )
        known = {
            str(Path(self.list_widget.item(i).data(Qt.ItemDataRole.UserRole)).resolve())
            .casefold()
            for i in range(self.list_widget.count())
        }
        for raw in paths:
            path = Path(raw)
            key = str(path.resolve()).casefold()
            if key in known:
                continue
            item = QListWidgetItem(f"{path.name}    {format_bytes(path.stat().st_size)}")
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.list_widget.addItem(item)
            known.add(key)
        self._update_count()

    def text(self) -> str:
        return self.text_edit.toPlainText()

    def remove_selected(self):
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))
        self._update_count()

    def paths(self) -> list[Path]:
        return [
            Path(self.list_widget.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self.list_widget.count())
        ]

    def clear(self):
        self.list_widget.clear()
        self.text_edit.clear()
        self._update_count()

    def _update_count(self):
        self.count_label.setText(f"{self.list_widget.count()} 个")
        self.list_widget.setVisible(self.list_widget.count() > 0)


class ProjectImportTab(QWidget):
    projects_changed = Signal()

    def __init__(self, store: ProjectStore):
        super().__init__()
        self.setMinimumSize(930, 640)
        self.store = store
        self.projects: list[dict] = []
        self._build()
        self.refresh_projects()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(18)

        form_card = card()
        form = QVBoxLayout(form_card)
        form.setContentsMargins(24, 22, 24, 22)
        form.setSpacing(15)
        form.addWidget(label("PROJECT IMPORT", "Eyebrow"))
        form.addWidget(label("导入小说章节", "SectionTitle"))
        form.addWidget(label("创建新项目，或把后续章节加入已有项目", "Muted"))

        mode_row = QHBoxLayout()
        self.new_radio = QRadioButton("新建项目")
        self.existing_radio = QRadioButton("加入已有项目")
        self.new_radio.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.new_radio)
        self.mode_group.addButton(self.existing_radio)
        self.new_radio.toggled.connect(self.sync_mode)
        mode_row.addWidget(self.new_radio)
        mode_row.addWidget(self.existing_radio)
        mode_row.addStretch()
        form.addLayout(mode_row)

        fields = QGridLayout()
        fields.setHorizontalSpacing(14)
        fields.setVerticalSpacing(10)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如：小说 A")
        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self.project_selected)
        self.source_combo = QComboBox()
        self.source_combo.addItems(LANGUAGE_OPTIONS)
        self.target_combo = QComboBox()
        self.target_combo.addItems(LANGUAGE_OPTIONS)
        self.target_combo.setCurrentText("日语")
        self.processing_mode_combo = QComboBox()
        self.processing_mode_combo.addItem("请选择处理模式", None)
        self.processing_mode_combo.addItems(TRANSLATION_MODES)
        self.rename_mode_combo = QComboBox()
        self.rename_mode_combo.addItem("开", True)
        self.rename_mode_combo.addItem("关", False)
        self.save_project_settings_button = QPushButton("保存项目设置")
        self.save_project_settings_button.clicked.connect(
            self.save_project_settings
        )
        self.categories_edit = QLineEdit("、".join(DEFAULT_CATEGORIES))
        fields.addWidget(label("项目名称"), 0, 0)
        fields.addWidget(self.name_edit, 0, 1)
        self.existing_label = label("已有项目")
        fields.addWidget(self.existing_label, 1, 0)
        fields.addWidget(self.project_combo, 1, 1)
        fields.addWidget(label("原文语言"), 2, 0)
        fields.addWidget(self.source_combo, 2, 1)
        fields.addWidget(label("目标语言"), 2, 2)
        fields.addWidget(self.target_combo, 2, 3)
        fields.addWidget(label("处理模式"), 3, 0)
        fields.addWidget(self.processing_mode_combo, 3, 1, 1, 3)
        fields.addWidget(label("专名本地化/改名"), 4, 0)
        fields.addWidget(self.rename_mode_combo, 4, 1, 1, 2)
        fields.addWidget(self.save_project_settings_button, 4, 3)
        fields.addWidget(label("替换类别"), 5, 0)
        fields.addWidget(self.categories_edit, 5, 1, 1, 3)
        fields.setColumnStretch(1, 1)
        fields.setColumnStretch(3, 1)
        form.addLayout(fields)
        form.addWidget(label("类别用逗号或顿号分隔；Skill 新增类别也可填写", "Hint"))

        uploads = QHBoxLayout()
        uploads.setSpacing(14)
        self.chapter_files = FileListBox(
            "TXT 章节（新项目必填）",
            "直接粘贴一章正文，或选择多个 TXT",
            "选择 TXT",
            "TXT 文本 (*.txt)",
        )
        self.skill_files = FileListBox(
            "规则 / Skill",
            "直接粘贴规则，或选择文件；翻译时自动附加",
            "选择规则文件",
            "规则文件 (*.md *.txt *.json *.yaml *.yml);;所有文件 (*.*)",
        )
        uploads.addWidget(self.chapter_files, 1)
        uploads.addWidget(self.skill_files, 1)
        form.addLayout(uploads)

        self.skill_management_widget = QGroupBox("已有 Skill 管理")
        skill_management = QHBoxLayout(self.skill_management_widget)
        skill_management.setContentsMargins(14, 20, 14, 14)
        skill_management.setSpacing(12)
        list_side = QVBoxLayout()
        list_side.addWidget(label("可多选后删除；编辑时只选一个", "Hint"))
        self.existing_skills = QListWidget()
        self.existing_skills.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.existing_skills.setMinimumHeight(120)
        self.existing_skills.itemSelectionChanged.connect(
            self.existing_skill_selection_changed
        )
        list_side.addWidget(self.existing_skills)
        delete_skill_button = QPushButton("删除选中")
        delete_skill_button.clicked.connect(self.delete_selected_skills)
        list_side.addWidget(delete_skill_button)
        editor_side = QVBoxLayout()
        editor_side.addWidget(label("选中一个 Skill 后可修改内容", "Hint"))
        self.existing_skill_editor = QTextEdit()
        self.existing_skill_editor.setPlaceholderText("请先在左侧选择一个 Skill")
        self.existing_skill_editor.setEnabled(False)
        self.existing_skill_editor.setMinimumHeight(120)
        editor_side.addWidget(self.existing_skill_editor)
        self.save_skill_button = QPushButton("保存修改")
        self.save_skill_button.setEnabled(False)
        self.save_skill_button.clicked.connect(self.save_selected_skill)
        editor_side.addWidget(self.save_skill_button)
        skill_management.addLayout(list_side, 1)
        skill_management.addLayout(editor_side, 1)
        form.addWidget(self.skill_management_widget)

        action_row = QHBoxLayout()
        action_row.addWidget(label("同内容章节不会重复导入", "Hint"))
        action_row.addStretch()
        self.import_button = QPushButton("创建项目并导入")
        self.import_button.setObjectName("Primary")
        self.import_button.clicked.connect(self.import_files)
        action_row.addWidget(self.import_button)
        form.addLayout(action_row)

        side = QVBoxLayout()
        status_card = card()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(18, 18, 18, 18)
        status_layout.addWidget(label("PROJECT STATUS", "Eyebrow"))
        self.status_title = label("等待导入", "CardTitle")
        self.status_text = label(
            "选择 TXT 后创建项目。每个项目会建立独立替换清单。", "Muted"
        )
        self.status_text.setWordWrap(True)
        status_layout.addWidget(self.status_title)
        status_layout.addWidget(self.status_text)
        side.addWidget(status_card)

        structure_card = card()
        structure = QVBoxLayout(structure_card)
        structure.setContentsMargins(18, 18, 18, 18)
        structure.addWidget(label("项目保存内容", "CardTitle"))
        for title_text, detail in (
            ("01  原文章节", "保留 TXT 原文件与导入记录"),
            ("02  统一替换清单", "按类别保存并保证原词唯一"),
            ("03  项目规则", "保存本小说专用 Skill"),
            ("04  译文与记录", "每次翻译单独留存版本"),
        ):
            structure.addWidget(label(title_text))
            structure.addWidget(label(detail, "Hint"))
        side.addWidget(structure_card)

        callout = QFrame()
        callout.setObjectName("Callout")
        callout_layout = QVBoxLayout(callout)
        callout_layout.addWidget(label("一致性保护", "CardTitle"))
        text = label(
            "“张三”首次确认为“夏目 二郎”后，后续章节只能复用该映射；"
            "模型若生成第二个名字，系统会拒绝覆盖并记录警告。",
            "Muted",
        )
        text.setWordWrap(True)
        callout_layout.addWidget(text)
        side.addWidget(callout)
        side.addStretch()

        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMinimumWidth(300)
        side_widget.setMaximumWidth(390)
        root.addWidget(form_card, 1)
        root.addWidget(side_widget)
        self.sync_mode()

    def refresh_projects(self, select_id: str | None = None):
        self.projects = self.store.list_projects()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for project in self.projects:
            self.project_combo.addItem(
                f"{project['name']}  ·  {len(project.get('chapters', []))} 章",
                project["id"],
            )
        if select_id:
            index = self.project_combo.findData(select_id)
            if index >= 0:
                self.project_combo.setCurrentIndex(index)
        self.project_combo.blockSignals(False)
        self.project_combo.setEnabled(bool(self.projects) and self.existing_radio.isChecked())
        if self.existing_radio.isChecked():
            self.project_selected()

    def current_project(self) -> dict | None:
        project_id = self.project_combo.currentData()
        return next((x for x in self.projects if x.get("id") == project_id), None)

    def sync_mode(self):
        is_new = self.new_radio.isChecked()
        self.name_edit.setEnabled(is_new)
        self.existing_label.setVisible(not is_new)
        self.project_combo.setVisible(not is_new)
        self.project_combo.setEnabled(not is_new and bool(self.projects))
        self.source_combo.setEnabled(is_new)
        self.target_combo.setEnabled(is_new)
        self.processing_mode_combo.setEnabled(is_new)
        self.rename_mode_combo.setEnabled(True)
        self.save_project_settings_button.setVisible(not is_new)
        self.import_button.setText("创建项目并导入" if is_new else "加入所选项目")
        self.skill_management_widget.setVisible(not is_new)
        if is_new:
            rename_index = self.rename_mode_combo.findData(True)
            self.rename_mode_combo.setCurrentIndex(max(0, rename_index))
            self.status_title.setText("新项目")
            self.status_text.setText(
                "导入时会创建项目资料、替换清单、规则目录和译文目录。"
            )
        else:
            self.project_selected()

    def project_selected(self):
        project = self.current_project()
        if not project:
            self.existing_skills.clear()
            self.status_title.setText("还没有项目")
            self.status_text.setText("请先新建项目并导入第一批 TXT。")
            return
        self.source_combo.setCurrentText(project.get("source_language", ""))
        self.target_combo.setCurrentText(project.get("target_language", ""))
        self.categories_edit.setText("、".join(project.get("categories", [])))
        rename_enabled = project.get("proper_noun_localization", True)
        rename_index = self.rename_mode_combo.findData(
            rename_enabled if isinstance(rename_enabled, bool) else True
        )
        self.rename_mode_combo.setCurrentIndex(max(0, rename_index))
        mode = project.get("processing_mode")
        mode_index = self.processing_mode_combo.findText(str(mode))
        if mode in TRANSLATION_MODES and mode_index >= 0:
            self.processing_mode_combo.setCurrentIndex(mode_index)
            self.processing_mode_combo.setEnabled(False)
        else:
            self.processing_mode_combo.setCurrentIndex(0)
            self.processing_mode_combo.setEnabled(True)
        chapters = project.get("chapters", [])
        characters = sum(item.get("characters", 0) for item in chapters)
        self.status_title.setText(project["name"])
        self.status_text.setText(
            f"已导入 {len(chapters)} 章，约 {characters:,} 字符。\n"
            f"替换类别 {len(project.get('categories', []))} 个，"
            f"规则文件 {len(project.get('skills', []))} 个。"
        )
        self.refresh_existing_skills(project["id"])

    def save_project_settings(self):
        project = self.current_project()
        if not project:
            QMessageBox.warning(self, "无法保存", "请选择已有项目。")
            return
        try:
            updated = self.store.update_project_settings(
                project["id"],
                proper_noun_localization=bool(
                    self.rename_mode_combo.currentData()
                ),
            )
            self.refresh_projects(updated["id"])
            self.projects_changed.emit()
            state = "开" if updated["proper_noun_localization"] else "关"
            QMessageBox.information(
                self,
                "保存完成",
                f"已保存到项目“{updated['name']}”：专名本地化/改名为{state}。",
            )
        except Exception as exc:
            QMessageBox.critical(self, "无法保存项目设置", str(exc))

    def refresh_existing_skills(self, project_id: str):
        self.existing_skills.clear()
        for skill in self.store.read_project_skills(project_id):
            suffix = "（文件缺失）" if skill.get("missing") else ""
            item = QListWidgetItem(f"{skill['name']}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, skill.get("id"))
            item.setData(Qt.ItemDataRole.UserRole + 1, skill.get("content", ""))
            item.setToolTip(skill.get("stored_name", ""))
            self.existing_skills.addItem(item)
        self.existing_skill_selection_changed()

    def existing_skill_selection_changed(self):
        selected = self.existing_skills.selectedItems()
        editable = len(selected) == 1 and "（文件缺失）" not in selected[0].text()
        self.existing_skill_editor.setEnabled(editable)
        self.save_skill_button.setEnabled(editable)
        self.existing_skill_editor.setPlainText(
            selected[0].data(Qt.ItemDataRole.UserRole + 1) if editable else ""
        )

    def save_selected_skill(self):
        selected = self.existing_skills.selectedItems()
        project = self.current_project()
        if len(selected) != 1 or not project:
            QMessageBox.warning(self, "无法编辑", "请只选择一个已有 Skill。")
            return
        try:
            self.store.update_skill(
                project["id"],
                selected[0].data(Qt.ItemDataRole.UserRole),
                self.existing_skill_editor.toPlainText(),
            )
            self.refresh_projects(project["id"])
            self.projects_changed.emit()
            QMessageBox.information(self, "保存完成", "Skill 已更新，可重新处理已有章节。")
        except Exception as exc:
            QMessageBox.critical(self, "无法保存 Skill", str(exc))

    def delete_selected_skills(self):
        selected = self.existing_skills.selectedItems()
        project = self.current_project()
        if not selected or not project:
            QMessageBox.warning(self, "无法删除", "请至少选择一个已有 Skill。")
            return
        pending_skill_count = len(self.store.pending_skill_analysis(project["id"]))
        skill_analysis_notice = (
            f"\n\n其中有 {pending_skill_count} 份新增或已修改的 Skill，将先各解析一次；"
            "未修改的 Skill 不会重复解析。"
            if pending_skill_count else ""
        )
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除选中的 {len(selected)} 个 Skill 吗？此操作会删除本地规则文件。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete_skills(
                project["id"],
                [item.data(Qt.ItemDataRole.UserRole) for item in selected],
            )
            self.refresh_projects(project["id"])
            self.projects_changed.emit()
            QMessageBox.information(self, "删除完成", "选中的 Skill 已删除。")
        except Exception as exc:
            QMessageBox.critical(self, "无法删除 Skill", str(exc))

    def import_files(self):
        try:
            chapter_paths = self.chapter_files.paths()
            chapter_text = self.chapter_files.text().strip()
            skill_text = self.skill_files.text().strip()
            skill_paths = self.skill_files.paths()
            if self.new_radio.isChecked() and not chapter_paths and not chapter_text:
                raise ValidationError("请选择 TXT 章节，或直接粘贴一章正文。")
            categories = split_categories(self.categories_edit.text())
            if self.new_radio.isChecked():
                processing_mode = self.processing_mode_combo.currentText()
                if processing_mode not in TRANSLATION_MODES:
                    raise ValidationError("请选择处理模式。")
                project = self.store.create_project(
                    self.name_edit.text(),
                    self.source_combo.currentText(),
                    self.target_combo.currentText(),
                    categories,
                    skill_paths,
                    processing_mode=processing_mode,
                    proper_noun_localization=bool(
                        self.rename_mode_combo.currentData()
                    ),
                )
            else:
                project = self.current_project()
                if not project:
                    raise ValidationError("请选择已有项目。")
                mode_was_unset = project.get("processing_mode") not in TRANSLATION_MODES
                if mode_was_unset:
                    processing_mode = self.processing_mode_combo.currentText()
                    if processing_mode not in TRANSLATION_MODES:
                        raise ValidationError("请先为旧项目选择一次处理模式。")
                    project = self.store.set_processing_mode(
                        project["id"], processing_mode
                    )
                categories_changed = any(
                    category not in project.get("categories", []) for category in categories
                )
                if not any(
                    (
                        chapter_paths,
                        chapter_text,
                        skill_paths,
                        skill_text,
                        categories_changed,
                        mode_was_unset,
                    )
                ):
                    raise ValidationError("没有可保存的内容。请选择 TXT、添加 Skill 或修改替换类别。")
                self.store.ensure_categories(project["id"], categories)
                self.store.add_skills(project["id"], skill_paths)
            if skill_text:
                self.store.add_skill_text(project["id"], skill_text)
            file_result = self.store.import_chapters(project["id"], chapter_paths)
            text_result = self.store.import_chapter_text(project["id"], chapter_text)
            imported_count = len(file_result.imported) + len(text_result.imported)
            skipped_count = len(file_result.skipped) + len(text_result.skipped)
            updated = self.store.load_project(project["id"])
            self.chapter_files.clear()
            self.skill_files.clear()
            self.name_edit.clear()
            self.refresh_projects(project["id"])
            self.status_title.setText("项目已更新")
            self.status_text.setText(
                f"{updated['name']}\n新增 {imported_count} 章，"
                f"现有 Skill {len(updated.get('skills', []))} 个。"
            )
            self.projects_changed.emit()
            QMessageBox.information(
                self,
                "保存完成",
                f"项目已更新；新增 {imported_count} 个章节。"
                + (
                    f"\n{skipped_count} 个重复章节已跳过。"
                    if skipped_count
                    else ""
                ),
            )
        except Exception as exc:
            QMessageBox.critical(self, "无法导入", str(exc))


class APIConfigTab(QWidget):
    profiles_changed = Signal()

    def __init__(self, store: APIConfigStore):
        super().__init__()
        self.setMinimumSize(900, 600)
        self.store = store
        self.profiles: list[dict] = []
        self.current_profile_id: str | None = None
        self.last_provider = "OpenAI"
        self.test_worker: ConnectionTestWorker | None = None
        self._build()
        self.refresh_profiles()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(18)
        form_card = card()
        form = QVBoxLayout(form_card)
        form.setContentsMargins(24, 22, 24, 22)
        form.setSpacing(14)
        form.addWidget(label("MODEL CONNECTION", "Eyebrow"))
        form.addWidget(label("API 与模型", "SectionTitle"))
        form.addWidget(label("保存多套配置，翻译时选择使用", "Muted"))

        profile_row = QHBoxLayout()
        profile_row.addWidget(label("配置"))
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self.profile_selected)
        profile_row.addWidget(self.profile_combo, 1)
        new_button = QPushButton("新建配置")
        new_button.clicked.connect(self.reset_form)
        profile_row.addWidget(new_button)
        form.addLayout(profile_row)

        fields = QGridLayout()
        fields.setHorizontalSpacing(14)
        fields.setVerticalSpacing(11)
        self.name_edit = QLineEdit("默认翻译模型")
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(PROVIDER_DEFAULTS.keys())
        self.provider_combo.currentTextChanged.connect(self.provider_changed)
        self.base_url_edit = QLineEdit(PROVIDER_DEFAULTS["OpenAI"]["base_url"])
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("填写服务商提供的准确模型 ID")
        self.key_edit = MaskedKeyLineEdit()
        self.key_edit.setPlaceholderText("输入 API Key")
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 600)
        self.timeout_spin.setSingleStep(10)
        self.timeout_spin.setValue(120)
        fields.addWidget(label("配置名称"), 0, 0)
        fields.addWidget(self.name_edit, 0, 1, 1, 3)
        fields.addWidget(label("API 服务商"), 1, 0)
        fields.addWidget(self.provider_combo, 1, 1)
        fields.addWidget(label("请求超时"), 1, 2)
        fields.addWidget(self.timeout_spin, 1, 3)
        fields.addWidget(label("API 地址"), 2, 0)
        fields.addWidget(self.base_url_edit, 2, 1, 1, 3)
        fields.addWidget(label("模型名称"), 3, 0)
        fields.addWidget(self.model_edit, 3, 1, 1, 3)
        fields.addWidget(label("API Key"), 4, 0)
        fields.addWidget(self.key_edit, 4, 1, 1, 3)
        fields.setColumnStretch(1, 1)
        fields.setColumnStretch(3, 1)
        form.addLayout(fields)
        self.key_hint = label("新建配置时必填", "Hint")
        form.addWidget(self.key_hint)
        form.addStretch()

        actions = QHBoxLayout()
        self.status_label = label("新配置尚未保存", "Hint")
        actions.addWidget(self.status_label)
        actions.addStretch()
        self.test_button = QPushButton("测试连接")
        self.test_button.clicked.connect(self.test_connection)
        save_button = QPushButton("保存 API 配置")
        save_button.setObjectName("Primary")
        save_button.clicked.connect(self.save_profile)
        actions.addWidget(self.test_button)
        actions.addWidget(save_button)
        form.addLayout(actions)

        side = QVBoxLayout()
        security = QFrame()
        security.setObjectName("SuccessCallout")
        security_layout = QVBoxLayout(security)
        security_layout.addWidget(label("Key 本地加密保存", "CardTitle"))
        security_text = label(
            "API Key 使用 Windows 当前用户加密后写入本机。"
            "界面重新载入配置时不会回显 Key，其他 Windows 账户无法直接解密。",
            "Muted",
        )
        security_text.setWordWrap(True)
        security_layout.addWidget(security_text)
        side.addWidget(security)

        info_card = card()
        info = QVBoxLayout(info_card)
        info.setContentsMargins(18, 18, 18, 18)
        info.addWidget(label("已支持的调用方式", "CardTitle"))
        for provider, detail in (
            ("OpenAI", "Responses API"),
            ("Anthropic", "Messages API"),
            ("Google Gemini", "Interactions API"),
            ("OpenAI 兼容接口", "Chat Completions 格式"),
        ):
            info.addWidget(label(provider))
            info.addWidget(label(detail, "Hint"))
        side.addWidget(info_card)
        note = QFrame()
        note.setObjectName("Callout")
        note_layout = QVBoxLayout(note)
        note_layout.addWidget(label("费用提醒", "CardTitle"))
        note_text = label(
            "“测试连接”和“开始翻译”都会实际调用所选服务商，可能产生 API 费用。",
            "Muted",
        )
        note_text.setWordWrap(True)
        note_layout.addWidget(note_text)
        side.addWidget(note)
        side.addStretch()
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMinimumWidth(300)
        side_widget.setMaximumWidth(390)
        root.addWidget(form_card, 1)
        root.addWidget(side_widget)

    def refresh_profiles(self, select_id: str | None = None):
        self.profiles = self.store.list_profiles()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("＋ 新建 API 配置", None)
        for profile in self.profiles:
            self.profile_combo.addItem(
                f"{profile['name']}  ·  {profile['provider']}", profile["id"]
            )
        if select_id:
            index = self.profile_combo.findData(select_id)
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)
        self.profile_combo.blockSignals(False)
        if select_id:
            self.profile_selected()

    def profile_selected(self):
        profile_id = self.profile_combo.currentData()
        if not profile_id:
            self.reset_form(keep_combo=True)
            return
        profile = next((x for x in self.profiles if x.get("id") == profile_id), None)
        if not profile:
            return
        self.current_profile_id = profile_id
        self.name_edit.setText(profile.get("name", ""))
        self.provider_combo.setCurrentText(profile.get("provider", ""))
        self.last_provider = profile.get("provider", "")
        self.base_url_edit.setText(profile.get("base_url", ""))
        self.model_edit.setText(profile.get("model", ""))
        self.key_edit.show_saved_mask()
        self.key_hint.setText("已保存的 Key 会以黑色遮蔽字符显示；如需更换，直接输入新 Key。")
        self.timeout_spin.setValue(profile.get("timeout_seconds", 120))
        self.status_label.setText("配置已载入")

    def reset_form(self, _checked=False, keep_combo=False):
        self.current_profile_id = None
        if not keep_combo:
            self.profile_combo.setCurrentIndex(0)
        self.name_edit.setText("默认翻译模型")
        self.provider_combo.setCurrentText("OpenAI")
        self.last_provider = "OpenAI"
        self.base_url_edit.setText(PROVIDER_DEFAULTS["OpenAI"]["base_url"])
        self.model_edit.clear()
        self.key_edit.clear_for_new_key()
        self.key_edit.setPlaceholderText("输入 API Key")
        self.key_hint.setText("新建配置时必填")
        self.timeout_spin.setValue(120)
        self.status_label.setText("新配置尚未保存")

    def provider_changed(self, provider: str):
        previous_default = PROVIDER_DEFAULTS.get(self.last_provider, {}).get(
            "base_url", ""
        )
        if not self.base_url_edit.text().strip() or (
            self.base_url_edit.text().strip() == previous_default
        ):
            self.base_url_edit.setText(
                PROVIDER_DEFAULTS.get(provider, {}).get("base_url", "")
            )
        self.last_provider = provider

    def form_profile(self, reveal_saved_key: bool = False) -> dict:
        api_key = "" if self.key_edit.has_saved_mask() else self.key_edit.text().strip()
        if not api_key and self.current_profile_id and reveal_saved_key:
            saved = self.store.get_profile(
                self.current_profile_id, reveal_key=True
            )
            api_key = saved.get("api_key", "") if saved else ""
        if not self.name_edit.text().strip():
            raise ValidationError("请输入配置名称。")
        if not self.base_url_edit.text().strip():
            raise ValidationError("请输入 API 地址。")
        if not self.model_edit.text().strip():
            raise ValidationError("请输入模型名称。")
        if not api_key:
            raise ValidationError("请输入 API Key。")
        return {
            "id": self.current_profile_id or "unsaved",
            "name": self.name_edit.text().strip(),
            "provider": self.provider_combo.currentText(),
            "base_url": self.base_url_edit.text().strip().rstrip("/"),
            "model": self.model_edit.text().strip(),
            "api_key": api_key,
            "timeout_seconds": self.timeout_spin.value(),
        }

    def entered_api_key(self) -> str | None:
        value = self.key_edit.text().strip()
        return None if not value or self.key_edit.has_saved_mask() else value

    def save_profile(self):
        try:
            result = self.store.upsert_profile(
                profile_id=self.current_profile_id,
                name=self.name_edit.text(),
                provider=self.provider_combo.currentText(),
                base_url=self.base_url_edit.text(),
                model=self.model_edit.text(),
                api_key=self.entered_api_key(),
                timeout_seconds=self.timeout_spin.value(),
            )
            self.current_profile_id = result["id"]
            self.key_edit.clear()
            self.refresh_profiles(result["id"])
            self.status_label.setText("配置已保存，Key 已加密")
            self.profiles_changed.emit()
            QMessageBox.information(self, "保存完成", "API 配置已安全保存。")
        except Exception as exc:
            QMessageBox.critical(self, "无法保存", str(exc))

    def test_connection(self):
        try:
            profile = self.form_profile(reveal_saved_key=True)
        except Exception as exc:
            QMessageBox.critical(self, "无法测试", str(exc))
            return
        answer = QMessageBox.question(
            self,
            "测试 API 连接",
            f"将向 {profile['provider']} 的 {profile['model']} 发送一条极短测试消息，"
            "可能产生少量 API 费用。是否继续？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.test_button.setEnabled(False)
        self.test_button.setText("正在测试…")
        self.test_worker = ConnectionTestWorker(profile)
        self.test_worker.succeeded.connect(self.test_succeeded)
        self.test_worker.failed.connect(self.test_failed)
        self.test_worker.start()

    def test_succeeded(self, result: str):
        self.test_button.setEnabled(True)
        self.test_button.setText("测试连接")
        self.status_label.setText("连接成功")
        QMessageBox.information(self, "连接成功", f"模型返回：{result[:300]}")

    def test_failed(self, error: str):
        self.test_button.setEnabled(True)
        self.test_button.setText("测试连接")
        self.status_label.setText("连接失败")
        QMessageBox.critical(self, "连接失败", error)


class ReplacementViewerDialog(QDialog):
    def __init__(self, chapter_name: str, rows: list[dict], parent=None):
        super().__init__(parent)
        self.rows = rows
        self.setWindowTitle(f"章节替换清单 · {chapter_name}")
        self.resize(820, 520)
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        toolbar.addWidget(label("替换类别", "Hint"))
        self.category_combo = QComboBox()
        self.category_combo.addItem("全部类别")
        self.category_combo.addItems(sorted({item["category"] for item in rows}))
        self.category_combo.currentTextChanged.connect(self.refresh_rows)
        toolbar.addWidget(self.category_combo)
        toolbar.addStretch()
        toolbar.addWidget(label(f"共 {len(rows)} 条", "Muted"))
        layout.addLayout(toolbar)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["类别", "原词", "新名称", "说明"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh_rows()

    def refresh_rows(self):
        category = self.category_combo.currentText()
        visible = [
            item for item in self.rows
            if category == "全部类别" or item["category"] == category
        ]
        self.table.setRowCount(len(visible))
        for row, item in enumerate(visible):
            for column, key in enumerate(("category", "source", "target", "notes")):
                self.table.setItem(row, column, QTableWidgetItem(str(item.get(key, ""))))


def available_name_candidates(store: ProjectStore, project_id: str) -> list[str]:
    """Return local pool names that are not already targets in this project."""
    pool_path = store.data_root / "name_pools.json"
    try:
        pools = json.loads(pool_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    used = {
        str(record.get("target", "")).strip()
        for registry in store.load_all_replacements(project_id).values()
        for record in registry.get("entries", {}).values()
    }
    candidates: list[str] = []
    for values in pools.values() if isinstance(pools, dict) else []:
        if not isinstance(values, list):
            continue
        for value in values:
            name = str(value).strip()
            if name and name not in used and name not in candidates:
                candidates.append(name)
    return candidates


def split_review_edits(edits: list[dict]) -> tuple[list[dict], list[str]]:
    """Separate no-cost local decisions from sources sent to automatic API review."""
    local = [item for item in edits if item.get("action") in {"ignore", "map"}]
    retries = [
        str(item.get("source", "")).strip()
        for item in edits if item.get("action") == "retry" and str(item.get("source", "")).strip()
    ]
    return local, retries


def format_automatic_review_summary(project: dict, *, cancelled: bool) -> str:
    """Describe the final persisted state after a whole-project API review."""
    chapters = list(project.get("chapters", []))
    translated = sum(item.get("status") == "已翻译" for item in chapters)
    review = sum(item.get("status") == "需要复核" for item in chapters)
    failed = sum(item.get("status") == "翻译失败" for item in chapters)
    prefix = "全书自动复核已取消" if cancelled else "全书自动复核结束"
    return (
        f"{prefix}：已翻译 {translated} 章；需要复核 {review} 章；"
        f"翻译失败 {failed} 章。"
    )


def format_automatic_review_report(report: dict | None, summary: str) -> str:
    """Make the automatic-review outcome explicit for the user."""
    if not report:
        return "API 实体复核没有返回可用结果。\n" + summary
    if not report.get("success"):
        return f"API 实体复核失败，未开始重译：{report.get('error', '未知错误')}\n{summary}"
    mappings = list(report.get("mappings", []))
    if not mappings:
        return (
            "API 全章润色重做已完成；每章均按自身完整原文和旧译文核对，"
            "未使用其他文件的同名映射。\n" + summary
        )
    samples = "、".join(
        f"{item.get('source', '')} → {item.get('target', '')}"
        for item in mappings[:12]
    )
    mapping_text = (
        f"自动新增或更新 {len(mappings)} 条名称映射"
        + (f"：{samples}" if samples else "。")
    )
    return f"API 实体复核完成；{mapping_text}\n{summary}"


def chapters_requiring_review(project: dict) -> list[str]:
    """Return only chapters whose latest saved translation still needs review."""
    return [
        str(chapter["id"])
        for chapter in project.get("chapters", [])
        if chapter.get("id")
        and chapter.get("translations")
        and chapter["translations"][-1].get("review_status") == "需要复核"
    ]


class ResidualReviewDialog(QDialog):
    """Collect per-residual local review decisions after a translation completes."""

    def __init__(self, store: ProjectStore, project: dict, chapter: dict,
                 residuals: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"原词复核 · {chapter.get('source_name', '章节')}")
        self.resize(760, min(640, 180 + len(residuals) * 110))
        self._rows: list[dict] = []
        layout = QVBoxLayout(self)
        layout.addWidget(label(
            "“用 API 重做”会自动复核实体、生成新名称并使用全文重译；会产生额外费用。", "Hint"
        ))
        categories = list(dict.fromkeys(
            list(project.get("categories", [])) + ["人名", "地名", "家族名", "组织名", "机构名", "种族名"]
        ))
        candidates = available_name_candidates(store, project["id"])
        for source in residuals:
            box = QGroupBox(f"原词：{source}")
            fields = QGridLayout(box)
            action = QComboBox()
            action.addItem("标记误报", "ignore")
            action.addItem("手动映射", "map")
            action.addItem("用 API 重做", "retry")
            category = QComboBox()
            category.addItems(categories)
            target = QLineEdit()
            target.setPlaceholderText("输入新的名称")
            notes = QLineEdit()
            notes.setPlaceholderText("角色身份、关系或出现语境（单字名使用 API 重做时必填）")
            pool = QComboBox()
            pool.addItem("从名字库选择（仅人名）", "")
            for name in candidates:
                pool.addItem(name, name)
            fields.addWidget(label("操作", "Hint"), 0, 0)
            fields.addWidget(action, 0, 1)
            fields.addWidget(label("类别", "Hint"), 0, 2)
            fields.addWidget(category, 0, 3)
            fields.addWidget(label("新名称", "Hint"), 1, 0)
            fields.addWidget(target, 1, 1, 1, 3)
            fields.addWidget(pool, 2, 1, 1, 3)
            fields.addWidget(label("角色 / 语境说明", "Hint"), 3, 0)
            fields.addWidget(notes, 3, 1, 1, 3)
            row = {"source": source, "action": action, "category": category,
                   "target": target, "notes": notes, "pool": pool}
            self._rows.append(row)
            action.currentIndexChanged.connect(lambda _=0, row=row: self._update_row(row))
            category.currentTextChanged.connect(lambda _="", row=row: self._update_row(row))
            pool.currentIndexChanged.connect(lambda _=0, row=row: self._choose_pool_name(row))
            self._update_row(row)
            layout.addWidget(box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self._accept_if_valid
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_row(self, row: dict) -> None:
        mapping = row["action"].currentData() == "map"
        row["category"].setEnabled(mapping)
        row["target"].setEnabled(mapping)
        row["notes"].setEnabled(mapping)
        row["pool"].setEnabled(mapping and row["category"].currentText() == "人名")

    def _choose_pool_name(self, row: dict) -> None:
        value = row["pool"].currentData()
        if value:
            row["target"].setText(str(value))

    def _accept_if_valid(self) -> None:
        for edit in self.edits():
            if edit["action"] == "map" and not edit["target"]:
                QMessageBox.warning(self, "缺少新名称", f"请为“{edit['source']}”填写新名称。")
                return
        self.accept()

    def edits(self) -> list[dict]:
        return [{
            "source": row["source"], "action": str(row["action"].currentData()),
            "category": row["category"].currentText(),
            "target": row["target"].text().strip(),
            "notes": row["notes"].text().strip(),
        } for row in self._rows]


class TranslationTab(QWidget):
    def __init__(self, store: ProjectStore, api_store: APIConfigStore):
        super().__init__()
        self.store = store
        self.api_store = api_store
        self.projects: list[dict] = []
        self.profiles: list[dict] = []
        self.worker: TranslationWorker | None = None
        self.row_by_chapter: dict[str, int] = {}
        self._restoring_processing_settings = False
        self._build()
        self.refresh_data()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(10)

        controls_card = card()
        controls = QVBoxLayout(controls_card)
        controls.setContentsMargins(16, 12, 16, 12)
        controls.setSpacing(8)
        heading = QHBoxLayout()
        heading_text = QVBoxLayout()
        heading_text.addWidget(label("TRANSLATION QUEUE", "Eyebrow"))
        heading_text.addWidget(label("翻译处理", "SectionTitle"))
        heading.addLayout(heading_text)
        heading.addStretch()
        self.processing_controls_toggle = QToolButton()
        self.processing_controls_toggle.setArrowType(Qt.ArrowType.UpArrow)
        self.processing_controls_toggle.setToolTip("收起翻译设置")
        self.processing_controls_toggle.setFixedSize(36, 36)
        self.processing_controls_toggle.clicked.connect(self.toggle_processing_controls)
        heading.addWidget(self.processing_controls_toggle)
        self.open_output_button = QToolButton()
        self.open_output_button.setText("打开译文目录")
        self.open_output_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.open_output_button.clicked.connect(self.open_output_folder)
        output_menu = QMenu(self.open_output_button)
        output_menu.addAction("打开全部译文目录", self.open_output_folder)
        output_menu.addAction("打开已完成译文目录", self.open_completed_output_folder)
        self.open_output_button.setMenu(output_menu)
        heading.addWidget(self.open_output_button)
        controls.addLayout(heading)

        self.processing_controls_body = QWidget()
        self._processing_controls_expanded = True
        processing_controls_layout = QVBoxLayout(self.processing_controls_body)
        processing_controls_layout.setContentsMargins(0, 0, 0, 0)
        processing_controls_layout.setSpacing(8)
        setup = QGridLayout()
        setup.setHorizontalSpacing(12)
        setup.setVerticalSpacing(9)
        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self.project_changed)
        self.profile_combo = QComboBox()
        self.mode_display = QLineEdit()
        self.mode_display.setReadOnly(True)
        self.mode_display.setEnabled(False)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1024, 128000)
        self.max_tokens.setSingleStep(1000)
        self.max_tokens.setValue(24000)
        setup.addWidget(label("小说项目"), 0, 0)
        setup.addWidget(self.project_combo, 0, 1)
        setup.addWidget(label("API 配置"), 0, 2)
        setup.addWidget(self.profile_combo, 0, 3)
        setup.addWidget(label("处理模式"), 1, 0)
        setup.addWidget(self.mode_display, 1, 1)
        setup.addWidget(label("最大输出 Token"), 1, 2)
        setup.addWidget(self.max_tokens, 1, 3)
        setup.setColumnStretch(1, 1)
        setup.setColumnStretch(3, 1)
        processing_controls_layout.addLayout(setup)
        extra_header = QHBoxLayout()
        extra_header.addWidget(label("本次额外要求（可选）", "Hint"))
        extra_header.addStretch()
        self.save_extra_instructions = QCheckBox("同时保存额外要求")
        extra_header.addWidget(self.save_extra_instructions)
        self.save_settings_button = QPushButton("保存项目配置")
        self.save_settings_button.clicked.connect(self.save_processing_settings)
        extra_header.addWidget(self.save_settings_button)
        processing_controls_layout.addLayout(extra_header)
        self.extra_instructions = QTextEdit()
        self.extra_instructions.setPlaceholderText(
            "例如：保留轻小说语气；钱二说话更冷淡；料理名称优先使用日本常见叫法……"
        )
        self.extra_instructions.setMaximumHeight(54)
        processing_controls_layout.addWidget(self.extra_instructions)
        controls.addWidget(self.processing_controls_body)
        root.addWidget(controls_card)

        self.queue_preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        queue_card = card()
        queue = QVBoxLayout(queue_card)
        queue.setContentsMargins(14, 14, 14, 14)
        queue_actions = QHBoxLayout()
        queue_actions.addWidget(label("章节队列", "CardTitle"))
        queue_actions.addStretch()
        pending_button = QPushButton("选择待处理")
        pending_button.clicked.connect(self.select_pending)
        all_button = QPushButton("全选")
        all_button.clicked.connect(self.toggle_all_checked)
        clear_button = QPushButton("清空")
        clear_button.clicked.connect(lambda: self.set_all_checked(False))
        self.view_replacements_button = QPushButton("查看替换清单")
        self.view_replacements_button.clicked.connect(self.view_chapter_replacements)
        self.delete_chapters_button = QPushButton("删除所选章节")
        self.delete_chapters_button.setObjectName("Danger")
        self.delete_chapters_button.clicked.connect(self.delete_selected_chapters)
        queue_actions.addWidget(pending_button)
        queue_actions.addWidget(all_button)
        queue_actions.addWidget(clear_button)
        queue_actions.addWidget(self.view_replacements_button)
        queue_actions.addWidget(self.delete_chapters_button)
        queue.addLayout(queue_actions)
        self.chapter_table = QTableWidget(0, 4)
        self.chapter_table.setHorizontalHeaderLabels(
            ["选择", "章节", "字符数", "状态"]
        )
        self.chapter_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.chapter_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.chapter_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.chapter_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chapter_table.customContextMenuRequested.connect(self.show_chapter_context_menu)
        self.chapter_table.verticalHeader().setVisible(False)
        header = self.chapter_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.chapter_table.itemSelectionChanged.connect(self.preview_selected)
        queue.addWidget(self.chapter_table)
        self.queue_preview_splitter.addWidget(queue_card)

        self.preview_card = card()
        preview = QVBoxLayout(self.preview_card)
        preview.setContentsMargins(14, 14, 14, 14)
        preview_header = QHBoxLayout()
        self.preview_title = label("章节预览", "CardTitle")
        preview_header.addWidget(self.preview_title)
        preview_header.addStretch()
        self.preview_toggle = QToolButton()
        self.preview_toggle.setArrowType(Qt.ArrowType.UpArrow)
        self.preview_toggle.setToolTip("收起预览")
        self.preview_toggle.setFixedSize(36, 36)
        self.preview_toggle.clicked.connect(self.toggle_preview)
        preview_header.addWidget(self.preview_toggle)
        preview.addLayout(preview_header)
        self.preview_body = QWidget()
        self._preview_expanded = True
        preview_body_layout = QVBoxLayout(self.preview_body)
        preview_body_layout.setContentsMargins(0, 0, 0, 0)
        preview_split = QSplitter(Qt.Orientation.Vertical)
        original_box = QWidget()
        original_layout = QVBoxLayout(original_box)
        original_layout.setContentsMargins(0, 0, 0, 0)
        original_layout.addWidget(label("原文", "Hint"))
        self.original_preview = QTextEdit()
        self.original_preview.setReadOnly(True)
        original_layout.addWidget(self.original_preview)
        translated_box = QWidget()
        translated_layout = QVBoxLayout(translated_box)
        translated_layout.setContentsMargins(0, 0, 0, 0)
        translated_layout.addWidget(label("最新译文", "Hint"))
        self.translation_preview = QTextEdit()
        self.translation_preview.setReadOnly(True)
        translated_layout.addWidget(self.translation_preview)
        preview_split.addWidget(original_box)
        preview_split.addWidget(translated_box)
        preview_body_layout.addWidget(preview_split)
        preview.addWidget(self.preview_body)
        self.queue_preview_splitter.addWidget(self.preview_card)
        self.queue_preview_splitter.setSizes([620, 500])
        footer_card = card()
        footer = QVBoxLayout(footer_card)
        footer.setContentsMargins(16, 13, 16, 13)
        self.progress_label = label("等待开始", "Hint")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        footer.addWidget(self.progress_label)
        footer.addWidget(self.progress_bar)
        bottom = QHBoxLayout()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(90)
        bottom.addWidget(self.log, 1)
        buttons = QVBoxLayout()
        self.start_button = QPushButton("开始处理")
        self.start_button.setObjectName("Primary")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_translation)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setObjectName("Danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_translation)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch()
        bottom.addLayout(buttons)
        footer.addLayout(bottom)
        self.content_splitter = QSplitter(Qt.Orientation.Vertical)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.addWidget(self.queue_preview_splitter)
        self.content_splitter.addWidget(footer_card)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 2)
        self.content_splitter.setSizes([430, 230])
        root.addWidget(self.content_splitter, 1)
        self.setMinimumHeight(620)

    def toggle_processing_controls(self) -> None:
        self._processing_controls_expanded = not self._processing_controls_expanded
        expanded = self._processing_controls_expanded
        self.processing_controls_body.setVisible(expanded)
        self.processing_controls_toggle.setArrowType(
            Qt.ArrowType.UpArrow if expanded else Qt.ArrowType.DownArrow
        )
        self.processing_controls_toggle.setToolTip(
            "收起翻译设置" if expanded else "展开翻译设置"
        )

    def toggle_preview(self) -> None:
        self._preview_expanded = not self._preview_expanded
        expanded = self._preview_expanded
        self.preview_body.setVisible(expanded)
        self.preview_title.setVisible(expanded)
        self.preview_card.setMaximumWidth(16_777_215 if expanded else 52)
        self.preview_toggle.setArrowType(
            Qt.ArrowType.UpArrow if expanded else Qt.ArrowType.LeftArrow
        )
        self.preview_toggle.setToolTip("收起预览" if expanded else "展开预览")
        if expanded:
            self.queue_preview_splitter.setSizes([620, 500])
        else:
            self.queue_preview_splitter.setSizes([1, 52])

    def refresh_data(self):
        selected_project = self.project_combo.currentData()
        self.projects = self.store.list_projects()
        for project in self.projects:
            for chapter in project.get("chapters", []):
                if chapter.get("status") == "需要复核":
                    TranslationEngine(self.store).refresh_residual_review(
                        project["id"], chapter["id"]
                    )
        self.projects = self.store.list_projects()
        self.profiles = self.api_store.list_profiles()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for project in self.projects:
            self.project_combo.addItem(project["name"], project["id"])
        if selected_project:
            index = self.project_combo.findData(selected_project)
            if index >= 0:
                self.project_combo.setCurrentIndex(index)
        self.project_combo.blockSignals(False)
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("请选择 API 配置", None)
        for profile in self.profiles:
            self.profile_combo.addItem(
                f"{profile['name']} · {profile['provider']}", profile["id"]
            )
        self.profile_combo.blockSignals(False)
        self.project_changed()

    def current_project(self) -> dict | None:
        project_id = self.project_combo.currentData()
        return next((x for x in self.projects if x.get("id") == project_id), None)

    def project_changed(self):
        project = self.current_project()
        self.chapter_table.setRowCount(0)
        self.row_by_chapter.clear()
        self.original_preview.clear()
        self.translation_preview.clear()
        if not project:
            self.mode_display.setText("未选择项目")
            self.start_button.setEnabled(False)
            self.save_settings_button.setEnabled(False)
            return
        self.save_settings_button.setEnabled(True)
        processing_mode = project.get("processing_mode")
        if processing_mode not in TRANSLATION_MODES:
            self.mode_display.setText("未设置（请到项目导入页选择）")
        else:
            self.mode_display.setText(processing_mode)
        self.restore_processing_settings(project)
        for row, chapter in enumerate(project.get("chapters", [])):
            self.chapter_table.insertRow(row)
            check_item = QTableWidgetItem()
            check_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            check_item.setCheckState(
                Qt.CheckState.Checked
                if chapter.get("status") != "已翻译"
                else Qt.CheckState.Unchecked
            )
            check_item.setData(Qt.ItemDataRole.UserRole, chapter["id"])
            self.chapter_table.setItem(row, 0, check_item)
            self.chapter_table.setItem(
                row, 1, QTableWidgetItem(chapter.get("source_name", ""))
            )
            self.chapter_table.setItem(
                row, 2, QTableWidgetItem(f"{chapter.get('characters', 0):,}")
            )
            self.chapter_table.setItem(
                row, 3, QTableWidgetItem(chapter.get("status", "待处理"))
            )
            self.row_by_chapter[chapter["id"]] = row
        if self.chapter_table.rowCount():
            self.chapter_table.selectRow(0)
        self.start_button.setEnabled(processing_mode in TRANSLATION_MODES)

    def restore_processing_settings(self, project: dict) -> None:
        settings = project.get("processing_settings", {})
        if not isinstance(settings, dict):
            settings = {}
        self._restoring_processing_settings = True
        controls = (
            self.profile_combo,
            self.max_tokens,
            self.extra_instructions,
            self.save_extra_instructions,
        )
        for control in controls:
            control.blockSignals(True)
        try:
            token_value = settings.get("max_output_tokens", 24000)
            valid_token = (
                type(token_value) is int
                and self.max_tokens.minimum()
                <= token_value
                <= self.max_tokens.maximum()
            )
            self.max_tokens.setValue(token_value if valid_token else 24000)
            profile_id = settings.get("profile_id")
            self.profile_combo.setItemText(0, "请选择 API 配置")
            self.profile_combo.setToolTip("")
            self.profile_combo.setCurrentIndex(0)
            profile_index = self.profile_combo.findData(profile_id)
            if profile_id and profile_index >= 0:
                self.profile_combo.setCurrentIndex(profile_index)
            elif profile_id:
                message = "已保存的 API 配置不可用，请重新选择并保存项目配置"
                self.profile_combo.setItemText(0, message)
                self.profile_combo.setToolTip(message)
            save_extra = settings.get("save_extra_instructions") is True
            self.save_extra_instructions.setChecked(save_extra)
            self.extra_instructions.setPlainText(
                str(settings.get("extra_instructions", "")) if save_extra else ""
            )
        finally:
            for control in controls:
                control.blockSignals(False)
            self._restoring_processing_settings = False

    def save_processing_settings(self, *_args) -> None:
        if self._restoring_processing_settings:
            return
        project = self.current_project()
        if not project:
            return
        save_extra = self.save_extra_instructions.isChecked()
        try:
            updated = self.store.update_processing_settings(project["id"], {
                "max_output_tokens": self.max_tokens.value(),
                "profile_id": self.profile_combo.currentData(),
                "save_extra_instructions": save_extra,
                "extra_instructions": (
                    self.extra_instructions.toPlainText().strip() if save_extra else ""
                ),
            })
        except Exception as exc:
            QMessageBox.warning(self, "无法保存项目配置", str(exc))
            return
        for index, existing in enumerate(self.projects):
            if existing.get("id") == updated.get("id"):
                self.projects[index] = updated
                break
        QMessageBox.information(
            self,
            "保存完成",
            f"API 配置和处理参数已保存到项目“{updated['name']}”。",
        )

    def selected_chapter_ids(self) -> list[str]:
        result = []
        for row in range(self.chapter_table.rowCount()):
            item = self.chapter_table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                result.append(item.data(Qt.ItemDataRole.UserRole))
        return result

    def set_all_checked(self, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.chapter_table.rowCount()):
            self.chapter_table.item(row, 0).setCheckState(state)

    def toggle_all_checked(self):
        has_rows = self.chapter_table.rowCount() > 0
        all_checked = has_rows and all(
            self.chapter_table.item(row, 0).checkState() == Qt.CheckState.Checked
            for row in range(self.chapter_table.rowCount())
        )
        self.set_all_checked(not all_checked)

    def select_pending(self):
        project = self.current_project()
        if not project:
            return
        statuses = {item["id"]: item.get("status") for item in project["chapters"]}
        for row in range(self.chapter_table.rowCount()):
            item = self.chapter_table.item(row, 0)
            chapter_id = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(
                Qt.CheckState.Unchecked
                if statuses.get(chapter_id) == "已翻译"
                else Qt.CheckState.Checked
            )

    def view_chapter_replacements(self):
        project = self.current_project()
        row = self.chapter_table.currentRow()
        if not project or row < 0:
            QMessageBox.warning(self, "无法查看", "请先高亮选择一个章节。")
            return
        item = self.chapter_table.item(row, 0)
        chapter_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not chapter_id:
            return
        chapter = self.store.get_chapter(project["id"], chapter_id)
        rows = self.store.chapter_replacements(project["id"], chapter_id)
        if not rows:
            QMessageBox.information(self, "章节替换清单", "本章节暂无替换记录。")
            return
        ReplacementViewerDialog(
            chapter.get("source_name", "章节"), rows, self
        ).exec()

    def chapter_context_menu_for_row(self, row: int) -> QMenu:
        menu = QMenu(self)
        project = self.current_project()
        item = self.chapter_table.item(row, 0) if row >= 0 else None
        chapter_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not project or not chapter_id:
            return menu
        return menu

    def show_chapter_context_menu(self, position) -> None:
        row = self.chapter_table.indexAt(position).row()
        menu = self.chapter_context_menu_for_row(row)
        if menu.actions():
            menu.exec(self.chapter_table.viewport().mapToGlobal(position))

    def open_residual_review(self, chapter_id: str, residuals: list[str] | None = None) -> None:
        project = self.current_project()
        if not project:
            return
        chapter = self.store.get_chapter(project["id"], chapter_id)
        if residuals is None:
            translations = chapter.get("translations", [])
            latest = translations[-1] if translations else {}
            if latest.get("review_status") != "需要复核":
                return
            residuals = list(latest.get("residual_sources", []))
        if not residuals:
            return
        row = self.row_by_chapter.get(chapter_id)
        dialog = ResidualReviewDialog(self.store, project, chapter, residuals, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        local_edits, retry_sources = split_review_edits(dialog.edits())
        if local_edits:
            try:
                reviewed = TranslationEngine(self.store).review_residuals(
                    project["id"], chapter_id, local_edits
                )
                if row == self.chapter_table.currentRow():
                    self.translation_preview.setPlainText(reviewed["translated_text"])
                if row is not None:
                    self.chapter_table.item(row, 3).setText(
                        reviewed["record"].get("review_status", "已翻译")
                    )
                self.log.append("  已完成本地复核，无需调用 API。")
            except Exception as exc:
                QMessageBox.warning(self, "无法保存复核", str(exc))
                return
        if retry_sources:
            self.pending_retries.append((chapter_id, retry_sources))
            if not self.worker or not self.worker.isRunning():
                queued_chapter_id, queued_sources = self.pending_retries.pop(0)
                self.start_residual_retry(queued_chapter_id, queued_sources)

    def delete_selected_chapters(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "无法删除", "处理任务进行中，不能删除章节。")
            return
        project = self.current_project()
        chapter_ids = self.selected_chapter_ids()
        if not project or not chapter_ids:
            QMessageBox.warning(self, "无法删除", "请先勾选要删除的章节。")
            return
        names = [
            item.get("source_name", "章节")
            for item in project.get("chapters", [])
            if item.get("id") in set(chapter_ids)
        ]
        answer = QMessageBox.question(
            self,
            "删除所选章节",
            f"确定删除所选 {len(chapter_ids)} 个章节吗？\n\n"
            + "\n".join(names[:10])
            + "\n\n原文、译文和处理记录都会一并删除，此操作无法撤销。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            deleted = self.store.delete_chapters(project["id"], chapter_ids)
        except Exception as exc:
            QMessageBox.critical(self, "删除失败", str(exc))
            return
        self.refresh_data()
        QMessageBox.information(self, "删除完成", f"已删除 {len(deleted)} 个章节。")

    def preview_selected(self):
        row = self.chapter_table.currentRow()
        project = self.current_project()
        if row < 0 or not project:
            return
        item = self.chapter_table.item(row, 0)
        if not item:
            return
        chapter_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            source = self.store.read_chapter_text(project["id"], chapter_id)
            translated = self.store.read_latest_translation(
                project["id"], chapter_id
            )
            self.original_preview.setPlainText(source)
            self.translation_preview.setPlainText(translated or "尚无译文")
        except Exception as exc:
            self.original_preview.setPlainText(str(exc))

    def start_translation(self):
        project = self.current_project()
        profile_id = self.profile_combo.currentData()
        chapter_ids = self.selected_chapter_ids()
        if not project:
            QMessageBox.warning(self, "无法开始", "请先导入小说项目。")
            return
        processing_mode = project.get("processing_mode")
        if processing_mode not in TRANSLATION_MODES:
            QMessageBox.warning(
                self, "无法开始", "请先到“项目导入”页为该项目选择一次处理模式。"
            )
            return
        if not profile_id:
            QMessageBox.warning(self, "无法开始", "请先保存一套 API 配置。")
            return
        if not chapter_ids:
            QMessageBox.warning(self, "无法开始", "请至少勾选一个章节。")
            return
        try:
            profile = self.api_store.get_profile(profile_id, reveal_key=True)
            if not profile or not profile.get("api_key"):
                raise ValidationError("所选 API 配置没有可用的 Key。")
        except Exception as exc:
            QMessageBox.critical(self, "无法读取 API 配置", str(exc))
            return
        pending_skill_count = len(self.store.pending_skill_analysis(project["id"]))
        skill_analysis_notice = (
            f"\n\n其中有 {pending_skill_count} 份新增或已修改的 Skill，将先各解析一次；"
            "未修改的 Skill 不会重复解析。"
            if pending_skill_count
            else ""
        )
        answer = QMessageBox.question(
            self,
            "开始翻译",
            f"即将把 {len(chapter_ids)} 个章节发送给 {profile['provider']} 的 "
            f"{profile['model']}。\n\n这会把所选小说正文、项目 Skill、替换清单和"
            f"本次额外要求（{len(self.extra_instructions.toPlainText().strip())} 字）"
            "发送给该服务商，并可能产生 API 费用；若返回格式异常，程序可能"
            "补发一次格式修复请求。" + skill_analysis_notice + "是否继续？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.store.mark_chapters_queued(project["id"], chapter_ids)
        for chapter_id in chapter_ids:
            row = self.row_by_chapter.get(chapter_id)
            if row is not None:
                chapter = self.store.get_chapter(project["id"], chapter_id)
                self.chapter_table.item(row, 3).setText(chapter.get("status", "待处理"))
                self.chapter_table.item(row, 3).setToolTip(chapter.get("last_error", ""))
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.project_combo.setEnabled(False)
        self.profile_combo.setEnabled(False)
        self.log.clear()
        self.progress_bar.setRange(0, len(chapter_ids))
        self.progress_bar.setValue(0)
        self.worker = TranslationWorker(
            self.store,
            project["id"],
            chapter_ids,
            profile,
            processing_mode,
            self.extra_instructions.toPlainText(),
            self.max_tokens.value(),
        )
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.chapter_started.connect(self.chapter_started)
        self.worker.chapter_succeeded.connect(self.chapter_succeeded)
        self.worker.chapter_failed.connect(self.chapter_failed)
        self.worker.log_message.connect(self.log.append)
        self.worker.task_finished.connect(self.translation_finished)
        self.worker.start()

    def cancel_translation(self):
        if self.worker:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            # The worker already owns a copy of the selected profile.  Let the
            # user prepare the next batch while an in-flight request unwinds.
            self.profile_combo.setEnabled(True)
            self.progress_label.setText("正在取消；当前网络请求结束后停止…")

    def update_progress(self, value: int, maximum: int, text: str):
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)

    def chapter_started(self, chapter_id: str):
        row = self.row_by_chapter.get(chapter_id)
        if row is not None:
            self.chapter_table.item(row, 3).setText("处理中")

    def chapter_succeeded(self, chapter_id: str, result: dict):
        row = self.row_by_chapter.get(chapter_id)
        if row is not None:
            self.chapter_table.item(row, 3).setText("已翻译")
            self.chapter_table.item(row, 0).setCheckState(Qt.CheckState.Unchecked)
        if row == self.chapter_table.currentRow():
            self.translation_preview.setPlainText(result["translated_text"])
        if result["warnings"]:
            self.log.append("  提示：" + "；".join(result["warnings"]))

    def chapter_failed(self, chapter_id: str, error: str):
        row = self.row_by_chapter.get(chapter_id)
        if row is not None:
            self.chapter_table.item(row, 3).setText("翻译失败")
            self.chapter_table.item(row, 3).setToolTip(error)

    def translation_finished(self, cancelled: bool):
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.project_combo.setEnabled(True)
        self.profile_combo.setEnabled(True)
        self.progress_label.setText("任务已取消" if cancelled else "本批次处理完成")
        current_id = self.project_combo.currentData()
        self.projects = self.store.list_projects()
        project = next((x for x in self.projects if x["id"] == current_id), None)
        if project:
            latest_status = {
                item["id"]: item.get("status", "待处理")
                for item in project.get("chapters", [])
            }
            for chapter_id, row in self.row_by_chapter.items():
                self.chapter_table.item(row, 3).setText(
                    latest_status.get(chapter_id, "待处理")
                )
        window = self.window()
        background = getattr(window, "_hidden_in_background", False)
        if not cancelled and background:
            window.tray_icon.showMessage(
                "小说本地化工作台",
                "后台翻译任务已完成。点击托盘图标可查看结果。",
            )
        elif not cancelled:
            QMessageBox.information(self, "翻译完成", "所选章节已处理完成。")

    def start_all_review_polish(self):
        project = self.current_project()
        if not project:
            QMessageBox.warning(self, "无法启动", "请先选择小说项目。")
            return
        chapter_ids = chapters_requiring_review(project)
        if not chapter_ids:
            QMessageBox.information(self, "无需润色", "当前项目没有需要复核的章节。")
            return
        residuals: list[str] = []
        for chapter in project.get("chapters", []):
            if str(chapter.get("id", "")) not in chapter_ids:
                continue
            latest = chapter.get("translations", [])[-1] if chapter.get("translations") else {}
            residuals.extend(str(item) for item in latest.get("residual_sources", []))
        answer = QMessageBox.question(
            self,
            "API 全章润色重做",
            f"将让 API 逐章重读并润色 {len(chapter_ids)} 个待复核章节，"
            "覆盖这些章节的旧译文，可能产生 API 费用。\n\n"
            "不会处理已完成章节，也不会复用其他文件的同名映射。是否继续？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.start_residual_retry("", list(dict.fromkeys(residuals)))

    def start_residual_retry(self, chapter_id: str, residuals: list[str]):
        project = self.current_project()
        profile_id = self.profile_combo.currentData()
        if not project or not profile_id:
            return
        profile = self.api_store.get_profile(profile_id, reveal_key=True)
        if not profile:
            QMessageBox.warning(self, "无法启动 API 重做", "请选择可用的 API 配置后重试。")
            return
        chapter_ids = chapters_requiring_review(project)
        if not chapter_ids:
            QMessageBox.information(self, "无需重做", "当前项目没有需要复核的章节。")
            return
        self.store.mark_chapters_queued(project["id"], chapter_ids)
        for queued_chapter_id in chapter_ids:
            row = self.row_by_chapter.get(queued_chapter_id)
            if row is not None:
                self.chapter_table.item(row, 3).setText("全书自动复核中")
        self.retry_in_progress = True
        self._automatic_review_report = None
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, len(chapter_ids))
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在启动 API 全章润色重做…")
        self.log.append("  已启动 API 全章润色重做。")
        instruction = (
            self.extra_instructions.toPlainText().strip()
            + "\n这是逐章完整润色重做。只根据本章原文和旧译文统一人名与称呼；"
            "不得采用其他文件的同名映射。此前误报词仅供核实："
            + "、".join(residuals)
        ).strip()
        self.worker = TranslationWorker(
            self.store, project["id"], chapter_ids, profile,
            project["processing_mode"], instruction, self.max_tokens.value(),
            automatic_review_sources=residuals,
        )
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.chapter_started.connect(self.chapter_started)
        self.worker.chapter_succeeded.connect(self.chapter_succeeded)
        self.worker.chapter_failed.connect(self.chapter_failed)
        self.worker.log_message.connect(self.log.append)
        self.worker.automatic_review_result.connect(self.automatic_review_reported)
        self.worker.task_finished.connect(self.retry_finished)
        self.worker.start()

    def automatic_review_reported(self, report: dict) -> None:
        self._automatic_review_report = dict(report)
        if report.get("success"):
            self.log.append("  已开始逐章全文润色；不会建立跨文件同名映射。")
        else:
            self.log.append(f"  API 实体复核失败：{report.get('error', '未知错误')}")

    def retry_finished(self, cancelled: bool):
        self.retry_in_progress = False
        self.worker = None
        self.cancel_button.setEnabled(False)
        self.project_combo.setEnabled(True)
        self.profile_combo.setEnabled(True)
        self.projects = self.store.list_projects()
        self.project_changed()
        project = self.current_project()
        if project:
            latest_project = self.store.load_project(project["id"])
            summary = format_automatic_review_summary(
                latest_project, cancelled=cancelled
            )
            report = format_automatic_review_report(
                self._automatic_review_report, summary
            )
            self.progress_label.setText(summary)
            self.log.append("  " + report.replace("\n", "；"))
            window = self.window()
            if getattr(window, "_hidden_in_background", False):
                window.tray_icon.showMessage("小说本地化工作台", report)
            else:
                QMessageBox.information(
                    self,
                    "自动复核已取消" if cancelled else "自动复核结果",
                    report,
                )
        if self.pending_retries and not cancelled:
            chapter_id, residuals = self.pending_retries.pop(0)
            self.start_residual_retry(chapter_id, residuals)

    def open_output_folder(self):
        project = self.current_project()
        if not project:
            QMessageBox.warning(self, "无法打开", "请先选择小说项目。")
            return
        path = self.store.project_dir(project["id"]) / "output"
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_completed_output_folder(self):
        project = self.current_project()
        if not project:
            QMessageBox.warning(self, "无法打开", "请先选择小说项目。")
            return
        try:
            path = self.store.refresh_completed_output(project["id"])
        except (OSError, ValidationError) as exc:
            QMessageBox.warning(self, "无法打开", f"无法整理已完成译文：{exc}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._hidden_in_background = False
        self.data_root = application_data_root()
        self.project_store = ProjectStore(self.data_root)
        self.api_store = APIConfigStore(self.data_root)
        self.setWindowTitle("小说本地化工作台")
        self.resize(1280, 850)
        self.setMinimumSize(1020, 700)
        self._build()
        self._build_tray_icon()

    def _build_tray_icon(self) -> None:
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        )
        self.tray_icon.setToolTip("小说本地化工作台")
        tray_menu = QMenu(self)
        show_action = tray_menu.addAction("打开工作台")
        show_action.triggered.connect(self.show_from_tray)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._tray_activated)

    def _tray_activated(self, reason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_from_tray()

    def show_from_tray(self) -> None:
        self._hidden_in_background = False
        self.tray_icon.hide()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _build(self):
        root = QWidget()
        root.setObjectName("AppRoot")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(76)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 10, 28, 10)
        mark = QLabel("文")
        mark.setObjectName("BrandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(44, 44)
        brand = QVBoxLayout()
        brand.setSpacing(1)
        brand.addWidget(label("小说本地化工作台", "BrandTitle"))
        brand.addWidget(label("长篇小说翻译与一致性管理", "BrandSub"))
        header_layout.addWidget(mark)
        header_layout.addSpacing(10)
        header_layout.addLayout(brand)
        header_layout.addStretch()
        header_layout.addWidget(label(f"本地桌面版  v{__version__}", "Version"))
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.project_tab = ProjectImportTab(self.project_store)
        self.api_tab = APIConfigTab(self.api_store)
        self.translation_tab = TranslationTab(self.project_store, self.api_store)
        project_scroll = QScrollArea()
        project_scroll.setWidgetResizable(True)
        project_scroll.setFrameShape(QFrame.Shape.NoFrame)
        project_scroll.setWidget(self.project_tab)
        api_scroll = QScrollArea()
        api_scroll.setWidgetResizable(True)
        api_scroll.setFrameShape(QFrame.Shape.NoFrame)
        api_scroll.setWidget(self.api_tab)
        self.tabs.addTab(project_scroll, "01  项目导入")
        self.tabs.addTab(api_scroll, "02  API 与 Key")
        self.tabs.addTab(self.translation_tab, "03  翻译处理")
        self.project_tab.projects_changed.connect(self.translation_tab.refresh_data)
        self.api_tab.profiles_changed.connect(self.translation_tab.refresh_data)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage(f"本机数据位置：{self.data_root}")

    def closeEvent(self, event):
        translation_worker = self.translation_tab.worker
        test_worker = self.api_tab.test_worker
        running = [
            worker
            for worker in (translation_worker, test_worker)
            if worker is not None and worker.isRunning()
        ]
        if not running:
            self.tray_icon.hide()
            event.accept()
            return
        self._hidden_in_background = True
        self.tray_icon.show()
        self.tray_icon.showMessage(
            "小说本地化工作台",
            "窗口已关闭，翻译任务正在后台继续处理。",
        )
        self.hide()
        event.ignore()


def _acquire_instance_lock(data_root: Path) -> QLockFile | None:
    data_root.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(data_root / ".novel_localizer.lock"))
    lock.setStaleLockTime(30_000)
    return lock if lock.tryLock(0) else None


def run_desktop_app():
    app = QApplication(sys.argv)
    app.setApplicationName("小说本地化工作台")
    app.setOrganizationName("NovelLocalizer")
    instance_lock = _acquire_instance_lock(application_data_root())
    if instance_lock is None:
        QMessageBox.information(
            None,
            "程序已在运行",
            "小说本地化工作台已经打开，可能隐藏在系统托盘中。",
        )
        return 0
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    exit_code = app.exec()
    instance_lock.unlock()
    return exit_code
