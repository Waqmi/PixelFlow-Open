from __future__ import annotations

import os
import queue
import shutil
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, QSettings, QTimer, QUrl, QSize, Qt
from PySide6.QtGui import QDesktopServices, QIcon, QImageReader, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from image_engine_api import RESOURCE_ROOT, SIZES, generate_images
from category_engine import CategoryTemplateManager, RuleCategoryClassifier, RuleSourceTypeClassifier, SourceTypeDecision
from local_model_assistant import LocalModelAssistant, find_onnx_model


APP_ICON_PATH = RESOURCE_ROOT / (
    "app-icon.icns" if sys.platform == "darwin" else "app-icon.ico"
)

LOGO_KEYS = (
    "logo_square_dark",
    "logo_square_light",
    "logo_tall_dark",
    "logo_tall_light",
)
DEFAULT_LOGO_FILES = {
    "logo_square_dark": "新logo 1440.png",
    "logo_square_light": "新logo 1440 2.png",
    "logo_tall_dark": "新logo 1920.png",
    "logo_tall_light": "新logo 1920 2.png",
}


def _user_data_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "PixelFlow"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PixelFlow"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "PixelFlow"


def _logo_storage_dir() -> Path:
    return _user_data_dir() / "brand-assets"


def _write_crash_report(error_traceback: str) -> Path | None:
    """Persist Python-level crash details where a user can retrieve them."""
    if sys.platform == "darwin":
        report_dir = Path.home() / "Library" / "Logs" / "PixelFlow"
    else:
        report_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PixelFlow" / "Crash Reports"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"crash-{datetime.now():%Y%m%d-%H%M%S}.txt"
        report_path.write_text(error_traceback, encoding="utf-8")
        return report_path
    except OSError:
        return None


def _install_crash_handler() -> None:
    original_hook = sys.excepthook

    def handle_exception(error_type, error, error_traceback) -> None:  # type: ignore[no-untyped-def]
        details = "".join(traceback.format_exception(error_type, error, error_traceback))
        report_path = _write_crash_report(details)
        location = f"\n\n崩溃报告：{report_path}" if report_path else ""
        try:
            QMessageBox.critical(
                None,
                "PixelFlow 意外退出",
                "应用发生未处理错误。请将崩溃报告发送给技术支持。"
                f"{location}\n\n错误摘要：{error}",
            )
        finally:
            original_hook(error_type, error, error_traceback)

    sys.excepthook = handle_exception


class ProductScaleSlider(QSlider):
    """Keep the confirmation dialog scrollable when the pointer is over a slider."""

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()


class ToggleOptionCard(QFrame):
    """Toggle the contained output option when its card is clicked."""

    def __init__(self, checkbox: QCheckBox) -> None:
        super().__init__()
        self._checkbox = checkbox
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._checkbox.toggle()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _LegacySellingPointsDialog(QDialog):
    """Collect short selling points used to label generated detail visuals."""

    def __init__(self, parent: QWidget, points: tuple[str, ...]) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑产品卖点")
        self.setMinimumWidth(480)
        self.setStyleSheet(parent.styleSheet())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        title = QLabel("产品卖点")
        title.setObjectName("settingsPageTitle")
        copy = QLabel("每行一个卖点。生成卖点图时，细节图会按顺序自动匹配；可在素材确认中调整分类。")
        copy.setObjectName("settingsDescription")
        copy.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(copy)
        self.editor = QTextEdit()
        self.editor.setPlainText("\n".join(points))
        self.editor.setPlaceholderText("例如：\n透气排汗\n防滑耐磨\n轻量支撑")
        self.editor.setMinimumHeight(180)
        layout.addWidget(self.editor)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存卖点")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selling_points(self) -> tuple[str, ...]:
        return tuple(
            line.strip() for line in self.editor.toPlainText().splitlines() if line.strip()
        )


class _LegacySettingsDialog(QDialog):
    """Brand-focused settings surface kept separate from the task flow."""

    def __init__(self, parent: QWidget, settings: QSettings) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("PixelFlow 设置")
        self.resize(960, 780)
        self.setMinimumSize(900, 740)
        self.setStyleSheet(parent.styleSheet())
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(16)
        header = QHBoxLayout()
        back = QPushButton("←")
        back.setObjectName("settingsBackButton")
        back.setToolTip("返回主界面")
        back.setFixedSize(58, 58)
        back.clicked.connect(self.reject)
        header.addWidget(back)
        title = QLabel("设置")
        title.setObjectName("settingsPageTitle")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        tabs_host = QFrame()
        tabs_host.setObjectName("settingsTabs")
        tabs = QHBoxLayout(tabs_host)
        tabs.setContentsMargins(7, 7, 7, 7)
        tabs.setSpacing(6)
        self.pages = QStackedWidget()
        self._add_settings_page(tabs, self.pages, "通用", self._general_page())
        self._add_settings_page(tabs, self.pages, "品牌", self._brand_page())
        self._add_settings_page(tabs, self.pages, "输出", self._output_page())
        self._add_settings_page(tabs, self.pages, "高级", self._advanced_page())
        layout.addWidget(tabs_host)
        layout.addWidget(self.pages, 1)

        footer = QHBoxLayout()
        reset = QPushButton("恢复默认品牌设置")
        reset.setObjectName("subtleButton")
        reset.clicked.connect(self._restore_brand_defaults)
        footer.addWidget(reset)
        footer.addStretch()
        cancel = QPushButton("取消")
        save = QPushButton("保存设置")
        save.setObjectName("startButton")
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self._save)
        footer.addWidget(cancel)
        footer.addWidget(save)
        layout.addLayout(footer)

    @staticmethod
    def _add_settings_page(
        tabs: QHBoxLayout,
        pages: QStackedWidget,
        name: str,
        widget: QWidget,
    ) -> None:
        button = QPushButton(name)
        button.setObjectName("settingsTabButton")
        button.setCheckable(True)
        button.setAutoExclusive(True)
        if not tabs.count():
            button.setChecked(True)
        tabs.addWidget(button, 1)
        scroll = QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        widget.setMinimumHeight(widget.sizeHint().height())
        scroll.setWidget(widget)
        pages.addWidget(scroll)
        button.clicked.connect(lambda _checked=False, index=pages.count() - 1: pages.setCurrentIndex(index))

    @staticmethod
    def _page_shell(eyebrow: str, title_text: str, copy: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        page.setObjectName("settingsPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 18, 0, 0)
        layout.setSpacing(15)
        eyebrow_label = QLabel(eyebrow)
        eyebrow_label.setObjectName("settingsEyebrow")
        title = QLabel(title_text)
        title.setObjectName("settingsPageTitle")
        description = QLabel(copy)
        description.setObjectName("settingsDescription")
        description.setWordWrap(True)
        layout.addWidget(eyebrow_label)
        layout.addWidget(title)
        layout.addWidget(description)
        return page, layout

    @staticmethod
    def _section_card(title_text: str, copy: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("settingsCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(11)
        title = QLabel(title_text)
        title.setObjectName("settingsCardTitle")
        layout.addWidget(title)
        if copy:
            hint = QLabel(copy)
            hint.setObjectName("settingsCardHint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
        return card, layout

    @staticmethod
    def _field_row(label_text: str, widget: QWidget, hint: str = "") -> QWidget:
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        label = QLabel(label_text)
        label.setObjectName("settingsFieldLabel")
        layout.addWidget(label)
        if hint:
            description = QLabel(hint)
            description.setObjectName("settingsFieldHint")
            description.setWordWrap(True)
            layout.addWidget(description)
        layout.addWidget(widget)
        return row

    @staticmethod
    def _segmented_control(labels: tuple[str, ...], selected: int = 0) -> QFrame:
        control = QFrame()
        control.setObjectName("segmentedControl")
        layout = QHBoxLayout(control)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        for index, label in enumerate(labels):
            button = QPushButton(label)
            button.setObjectName("segmentButton")
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setChecked(index == selected)
            layout.addWidget(button, 1)
        return control

    def _general_page(self) -> QWidget:
        page, layout = self._page_shell(
            "WORKFLOW DEFAULTS",
            "把常用操作变成默认值",
            "这些设置会在每次新任务开始时作为初始状态加载，当前任务仍可临时调整。",
        )
        layout.setSpacing(14)
        card, card_layout = self._section_card("输出行为", "决定每次生成完成后文件的默认去向。")
        card_layout.addWidget(
            self._field_row(
                "默认输出位置",
                self._segmented_control(("跟随素材文件夹", "上次使用位置", "固定输出目录")),
            )
        )
        open_output = QCheckBox("生成完成后自动打开输出文件夹")
        open_output.setChecked(self.settings.value("open_output", True, type=bool))
        self.open_output = open_output
        card_layout.addWidget(open_output)
        layout.addWidget(card)

        card, card_layout = self._section_card("界面偏好", "不影响图片生成结果。")
        card_layout.addWidget(
            self._field_row("外观主题", self._segmented_control(("浅色", "深色", "跟随系统"), 2))
        )
        card_layout.addWidget(
            self._field_row("界面语言", self._segmented_control(("简体中文", "繁體中文", "English")))
        )
        layout.addWidget(card)
        layout.addStretch()
        return page

    def _brand_page(self) -> QWidget:
        page, layout = self._page_shell(
            "BRAND ASSETS",
            "品牌在每张图上的一致呈现",
            "为方图和竖图分别准备深色、浅色 Logo。PixelFlow 会按背景明暗自动选择版本。",
        )
        card, card_layout = self._section_card("Logo 资源", "四张文件均为完整画布 PNG，不是裁切的小图标。")
        card_layout.addWidget(
            self._logo_asset_group(
                "方图 Logo",
                "1440 × 1440",
                "方图深色 Logo",
                "方图浅色 Logo",
            )
        )
        card_layout.addWidget(
            self._logo_asset_group(
                "竖图 Logo",
                "1440 × 1920",
                "竖图深色 Logo",
                "竖图浅色 Logo",
            )
        )
        layout.addWidget(card)
        rule_card, rule_layout = self._section_card("上传规范")
        rule = QLabel("PNG · RGBA 透明背景 · 完整尺寸画布 · 不带白底、边框、阴影或额外文字")
        rule.setObjectName("settingsRule")
        rule.setWordWrap(True)
        rule_layout.addWidget(rule)
        layout.addWidget(rule_card)
        layout.addStretch()
        return page

    def _logo_asset_group(
        self,
        title_text: str,
        ratio: str,
        dark_name: str,
        light_name: str,
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("logoAssetGroup")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(13, 11, 13, 11)
        layout.setSpacing(11)
        label_column = QVBoxLayout()
        label = QLabel(title_text)
        label.setObjectName("logoGroupTitle")
        label_column.addWidget(label)
        meta = QLabel(ratio)
        meta.setObjectName("logoGroupMeta")
        label_column.addWidget(meta)
        label_column.addStretch()
        layout.addLayout(label_column, 0)
        layout.addWidget(self._logo_variant(dark_name, False), 1)
        layout.addWidget(self._logo_variant(light_name, True), 1)
        return card

    def _logo_variant(self, name: str, dark: bool) -> QFrame:
        card = QFrame()
        card.setObjectName("logoVariantDark" if dark else "logoVariantLight")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(9, 8, 9, 8)
        layout.setSpacing(8)
        preview = QLabel("PF")
        preview.setObjectName("logoPreviewDark" if dark else "logoPreviewLight")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setFixedSize(40, 40)
        layout.addWidget(preview)
        info = QVBoxLayout()
        name_label = QLabel("浅色背景" if not dark else "深色背景")
        name_label.setObjectName("logoVariantName")
        info.addWidget(name_label)
        detail = QLabel("深色 Logo" if not dark else "浅色 Logo")
        detail.setObjectName("logoVariantMeta")
        info.addWidget(detail)
        layout.addLayout(info, 1)
        replace = QPushButton("替换")
        replace.setObjectName("logoReplaceButton")
        replace.clicked.connect(lambda _checked=False, label=name: self._choose_logo(label))
        layout.addWidget(replace)
        return card

    def _output_page(self) -> QWidget:
        page, layout = self._page_shell(
            "OUTPUT PRESETS",
            "让每次导出从正确规格开始",
            "这些默认项会预先选中，不限制当前任务重新选择规格。",
        )
        card, card_layout = self._section_card("默认输出规格")
        size_layout = QHBoxLayout()
        for size in SIZES:
            check = QCheckBox(size)
            check.setChecked(True)
            size_layout.addWidget(check)
        card_layout.addLayout(size_layout)
        layout.addWidget(card)
        card, card_layout = self._section_card("默认生成内容", "作为新任务的初始状态。")
        choices = QGridLayout()
        for index, (label, checked) in enumerate((
            ("Logo", True),
            ("自动识别素材类型", True),
            ("唯品专享图", False),
            ("自动生成模特主图", True),
            ("自动生成卖点图", False),
        )):
            check = QCheckBox(label)
            check.setChecked(checked)
            choices.addWidget(check, index // 2, index % 2)
        card_layout.addLayout(choices)
        layout.addWidget(card)
        layout.addStretch()
        return page

    def _advanced_page(self) -> QWidget:
        page, layout = self._page_shell(
            "ADVANCED CONTROLS",
            "将复杂控制留在需要的时候",
            "品类比例、文件大小限制等当前任务参数仍在主界面的高级设置内；这里为后续全局默认值预留。",
        )
        card, card_layout = self._section_card("当前版本", "高级全局默认值将在规则稳定后开放，避免日常操作增加负担。")
        badge = QLabel("保持自动化优先")
        badge.setObjectName("settingsBadge")
        card_layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(card)
        layout.addStretch()
        return page

    def _choose_logo(self, label: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"选择{label}", "", "PNG 图片 (*.png)")
        if path:
            QMessageBox.information(self, "Logo 检查", f"已选择：{Path(path).name}\n\n保存前会检查 PNG、透明通道和画布尺寸。")

    def _restore_brand_defaults(self) -> None:
        QMessageBox.information(self, "已恢复默认", "品牌 Logo 将恢复为 PixelFlow 内置资源。")

    def _save(self) -> None:
        self.settings.setValue("open_output", self.open_output.isChecked())
        self.accept()


class _LegacyV15SettingsDialog(QDialog):
    """Compact settings surface for brand assets and the local model."""

    def __init__(self, parent: QWidget, settings: QSettings) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("PixelFlow 设置")
        self.resize(980, 760)
        self.setMinimumSize(860, 680)
        self.setStyleSheet(parent.styleSheet())
        self.logo_paths: dict[str, Path] = {}
        self.logo_previews: dict[str, QLabel] = {}
        self.logo_filenames: dict[str, QLabel] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(16)
        header = QHBoxLayout()
        back = QPushButton("←")
        back.setObjectName("settingsBackButton")
        back.setToolTip("返回主界面")
        back.setFixedSize(52, 52)
        back.clicked.connect(self.reject)
        header.addWidget(back)
        title = QLabel("设置")
        title.setObjectName("settingsPageTitle")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        tabs_host = QFrame()
        tabs_host.setObjectName("settingsTabs")
        tabs = QHBoxLayout(tabs_host)
        tabs.setContentsMargins(7, 7, 7, 7)
        tabs.setSpacing(6)
        self.pages = QStackedWidget()
        self._add_page(tabs, "品牌设置", self._brand_page())
        self._add_page(tabs, "模特图增强", self._model_page())
        layout.addWidget(tabs_host)
        layout.addWidget(self.pages, 1)

        footer = QHBoxLayout()
        reset = QPushButton("恢复默认品牌设置")
        reset.setObjectName("subtleButton")
        reset.clicked.connect(self._restore_brand_defaults)
        footer.addWidget(reset)
        footer.addStretch()
        cancel = QPushButton("取消")
        save = QPushButton("保存设置")
        save.setObjectName("startButton")
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self._save)
        footer.addWidget(cancel)
        footer.addWidget(save)
        layout.addLayout(footer)

    def _add_page(self, tabs: QHBoxLayout, name: str, widget: QWidget) -> None:
        button = QPushButton(name)
        button.setObjectName("settingsTabButton")
        button.setCheckable(True)
        button.setAutoExclusive(True)
        index = self.pages.count()
        button.setChecked(index == 0)
        tabs.addWidget(button, 1)
        scroll = QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        self.pages.addWidget(scroll)
        button.clicked.connect(lambda _checked=False, page_index=index: self.pages.setCurrentIndex(page_index))

    @staticmethod
    def _page_shell(title_text: str, copy: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        page.setObjectName("settingsPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 18, 0, 0)
        layout.setSpacing(15)
        title = QLabel(title_text)
        title.setObjectName("settingsPageTitle")
        description = QLabel(copy)
        description.setObjectName("settingsDescription")
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)
        return page, layout

    @staticmethod
    def _section_card(title_text: str, copy: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("settingsCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(11)
        title = QLabel(title_text)
        title.setObjectName("settingsCardTitle")
        layout.addWidget(title)
        if copy:
            hint = QLabel(copy)
            hint.setObjectName("settingsCardHint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
        return card, layout

    def _brand_page(self) -> QWidget:
        page, layout = self._page_shell(
            "品牌资源",
            "让每张主图使用正确的品牌 Logo",
        )
        card, card_layout = self._section_card("Logo 资源", "预览使用当前实际 Logo，浅色和深色版本分别对应相同背景。")
        card_layout.addWidget(self._logo_group("方图 Logo", "1440 × 1440", "logo_square_dark", "logo_square_light"))
        card_layout.addWidget(self._logo_group("竖图 Logo", "1440 × 1920", "logo_tall_dark", "logo_tall_light"))
        layout.addWidget(card)
        rule_card, rule_layout = self._section_card("上传规范")
        rule = QLabel("PNG · RGBA 透明背景 · 完整尺寸画布 · 不带白底、边框、阴影或额外文字")
        rule.setObjectName("settingsRule")
        rule.setWordWrap(True)
        rule_layout.addWidget(rule)
        layout.addWidget(rule_card)
        layout.addStretch()
        return page

    def _logo_group(self, title_text: str, ratio: str, dark_key: str, light_key: str) -> QFrame:
        group = QFrame()
        group.setObjectName("logoAssetGroup")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(13, 11, 13, 11)
        layout.setSpacing(8)
        heading = QHBoxLayout()
        title = QLabel(title_text)
        title.setObjectName("logoGroupTitle")
        heading.addWidget(title)
        meta = QLabel(ratio)
        meta.setObjectName("logoGroupMeta")
        heading.addWidget(meta)
        heading.addStretch()
        layout.addLayout(heading)
        previews = QHBoxLayout()
        previews.setSpacing(10)
        previews.addWidget(self._logo_variant(dark_key, "浅色背景", "深色 Logo", False), 1)
        previews.addWidget(self._logo_variant(light_key, "深色背景", "浅色 Logo", True), 1)
        layout.addLayout(previews)
        return group

    def _current_logo_path(self, key: str) -> Path:
        saved = str(self.settings.value(f"{key}_path", "") or "")
        if saved and Path(saved).is_file():
            return Path(saved)
        return RESOURCE_ROOT / DEFAULT_LOGO_FILES[key]

    @staticmethod
    def _update_preview(preview: QLabel, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            preview.setText("无法预览")
            return
        target = preview.size()
        if target.width() <= 0 or target.height() <= 0:
            target = preview.sizeHint()
        preview.setPixmap(pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _logo_variant(self, key: str, background: str, logo_name: str, dark_background: bool) -> QFrame:
        card = QFrame()
        card.setObjectName("logoVariantDark" if dark_background else "logoVariantLight")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(7)
        preview = QLabel()
        preview.setObjectName("logoPreviewDark" if dark_background else "logoPreviewLight")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setMinimumHeight(130)
        preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        path = self._current_logo_path(key)
        self.logo_paths[key] = path
        self.logo_previews[key] = preview
        self._update_preview(preview, path)
        layout.addWidget(preview)
        label_row = QHBoxLayout()
        name = QLabel(background)
        name.setObjectName("logoVariantName")
        label_row.addWidget(name)
        detail = QLabel(logo_name)
        detail.setObjectName("logoVariantMeta")
        label_row.addWidget(detail)
        label_row.addStretch()
        layout.addLayout(label_row)
        filename = QLabel(path.name)
        filename.setObjectName("logoVariantMeta")
        filename.setWordWrap(True)
        layout.addWidget(filename)
        self.logo_filenames[key] = filename
        replace = QPushButton("更换")
        replace.setObjectName("logoReplaceButton")
        replace.clicked.connect(lambda _checked=False, logo_key=key: self._choose_logo(logo_key, preview, filename, logo_name))
        layout.addWidget(replace, 0, Qt.AlignmentFlag.AlignRight)
        return card

    def _choose_logo(self, key: str, preview: QLabel, filename: QLabel, logo_name: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"选择{logo_name}", "", "PNG 图片 (*.png)")
        if not path:
            return
        selected = Path(path)
        expected = (1440, 1440) if "square" in key else (1440, 1920)
        reader = QImageReader(str(selected))
        if bytes(reader.format()).lower() != b"png" or reader.size().width() != expected[0] or reader.size().height() != expected[1]:
            QMessageBox.warning(self, "Logo 检查失败", f"需要 PNG，画布尺寸必须为 {expected[0]} × {expected[1]}。")
            return
        image = QPixmap(str(selected)).toImage()
        if image.isNull() or not image.hasAlphaChannel():
            QMessageBox.warning(self, "Logo 检查失败", "Logo 必须是可读取且包含透明通道的 PNG。")
            return
        try:
            storage = _logo_storage_dir()
            storage.mkdir(parents=True, exist_ok=True)
            destination = storage / DEFAULT_LOGO_FILES[key]
            shutil.copy2(selected, destination)
        except OSError as error:
            QMessageBox.warning(self, "Logo 保存失败", str(error))
            return
        self.logo_paths[key] = destination
        self._update_preview(preview, destination)
        filename.setText(destination.name)

    def _model_page(self) -> QWidget:
        page, layout = self._page_shell(
            "模特图增强",
            "模型只辅助判断人物主体和安全构图，不生成新图片。不可用时自动回退规则。",
        )
        card, card_layout = self._section_card("本地模型", "模型放在 App 外部，便于跨平台更新和回退。")
        self.local_model_check = QCheckBox("启用本地模型辅助判断")
        self.local_model_check.setChecked(self.settings.value("local_model_enabled", False, type=bool))
        card_layout.addWidget(self.local_model_check)
        self.local_model_path = QLineEdit(str(self.settings.value("local_model_path", "") or ""))
        self.local_model_path.setReadOnly(True)
        self.local_model_path.setPlaceholderText("尚未选择模型文件夹")
        row = QHBoxLayout()
        row.addWidget(self.local_model_path, 1)
        choose = QPushButton("选择文件夹")
        choose.clicked.connect(self._choose_model_dir)
        row.addWidget(choose)
        card_layout.addLayout(row)
        self.local_model_status = QLabel()
        self.local_model_status.setObjectName("settingsFieldHint")
        card_layout.addWidget(self.local_model_status)
        test = QPushButton("测试模型")
        test.setObjectName("subtleButton")
        test.clicked.connect(self._test_model)
        card_layout.addWidget(test, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(card)
        layout.addStretch()
        self._refresh_model_status()
        return page

    def _choose_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择本地模型文件夹")
        if path:
            self.local_model_path.setText(path)
            self._refresh_model_status()

    def _refresh_model_status(self) -> None:
        path = Path(self.local_model_path.text().strip())
        available, message = LocalModelAssistant.inspect(path)
        self.local_model_status.setText(f"状态：{message}")

    def _test_model(self) -> None:
        self._refresh_model_status()
        if "模型可用" in self.local_model_status.text():
            QMessageBox.information(self, "模型检查", "已检测到 ONNX 模型文件。")
        else:
            QMessageBox.warning(self, "模型检查", self.local_model_status.text().replace("状态：", ""))

    def _restore_brand_defaults(self) -> None:
        answer = QMessageBox.question(self, "恢复默认品牌设置", "四个 Logo 将恢复为 PixelFlow 内置资源。继续吗？")
        if answer != QMessageBox.StandardButton.Yes:
            return
        for key in LOGO_KEYS:
            self.settings.remove(f"{key}_path")
            self.logo_paths[key] = RESOURCE_ROOT / DEFAULT_LOGO_FILES[key]
            self._update_preview(self.logo_previews[key], self.logo_paths[key])
            self.logo_filenames[key].setText(self.logo_paths[key].name)
        QMessageBox.information(self, "已恢复默认", "品牌 Logo 已恢复为内置资源。请点击“保存设置”完成保存。")

    def _save(self) -> None:
        for key, path in self.logo_paths.items():
            self.settings.setValue(f"{key}_path", str(path) if path != RESOURCE_ROOT / DEFAULT_LOGO_FILES[key] else "")
        self.settings.setValue("local_model_enabled", self.local_model_check.isChecked())
        self.settings.setValue("local_model_path", self.local_model_path.text().strip())
        self.accept()


class V15SettingsDialog(_LegacyV15SettingsDialog):
    """Compact cc-switch-inspired settings without nested scrolling cards."""

    OUTPUT_OPTIONS = (
        ("include_logo", "生成含 Logo / 无 Logo", "在主页显示 Logo 输出开关"),
        ("include_vip", "生成唯品专享 1440", "在主页显示唯品图输出开关"),
        ("enable_material_understanding", "启用素材理解层（规则识别）", "在主页显示素材分类开关"),
        ("include_model_images", "自动生成模特主图", "模特图开关仅在设置页控制"),
    )

    def __init__(self, parent: QWidget, settings: QSettings) -> None:
        super().__init__(parent, settings)
        self.resize(860, 620)
        self.setMinimumSize(820, 620)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(12)
        header = QHBoxLayout()
        back = QPushButton("←")
        back.setObjectName("settingsBackButton")
        back.setToolTip("返回主界面")
        back.setFixedSize(42, 42)
        back.clicked.connect(self.reject)
        header.addWidget(back)
        title = QLabel("设置")
        title.setObjectName("settingsPageTitle")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        tabs_host = QFrame()
        tabs_host.setObjectName("settingsTabs")
        tabs = QHBoxLayout(tabs_host)
        tabs.setContentsMargins(5, 5, 5, 5)
        tabs.setSpacing(4)
        self.pages = QStackedWidget()
        self._add_flat_page(tabs, "品牌设置", self._flat_brand_page())
        self._add_flat_page(tabs, "输出选项", self._output_options_page())
        self._add_flat_page(tabs, "模特图增强", self._flat_model_page())
        self._add_flat_page(tabs, "关于", self._flat_about_page())
        layout.addWidget(tabs_host)
        layout.addWidget(self.pages, 1)

        footer = QHBoxLayout()
        reset = QPushButton("恢复默认品牌设置")
        reset.setObjectName("subtleButton")
        reset.clicked.connect(self._restore_brand_defaults)
        footer.addWidget(reset)
        footer.addStretch()
        cancel = QPushButton("取消")
        save = QPushButton("保存设置")
        save.setObjectName("startButton")
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self._save)
        footer.addWidget(cancel)
        footer.addWidget(save)
        layout.addLayout(footer)
        QTimer.singleShot(0, self._refresh_logo_previews)

    def _refresh_logo_previews(self) -> None:
        for key, preview in self.logo_previews.items():
            self._update_preview(preview, self.logo_paths.get(key, Path()))

    def _add_flat_page(self, tabs: QHBoxLayout, name: str, page: QWidget) -> None:
        button = QPushButton(name)
        button.setObjectName("settingsTabButton")
        button.setCheckable(True)
        button.setAutoExclusive(True)
        index = self.pages.count()
        button.setChecked(index == 0)
        tabs.addWidget(button, 1)
        self.pages.addWidget(page)
        button.clicked.connect(lambda _checked=False, page_index=index: self.pages.setCurrentIndex(page_index))
        self.pages.currentChanged.connect(
            lambda page_index, tab_index=index, tab_button=button: tab_button.setChecked(page_index == tab_index)
        )

    @staticmethod
    def _flat_page(title_text: str, description_text: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        page.setObjectName("settingsPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 0)
        layout.setSpacing(10)
        title = QLabel(title_text)
        title.setObjectName("settingsPageTitle")
        description = QLabel(description_text)
        description.setObjectName("settingsDescription")
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)
        return page, layout

    def _flat_brand_page(self) -> QWidget:
        page, layout = self._flat_page("品牌资源", "四个 Logo 直接显示真实预览；深色 Logo 对应浅色背景，浅色 Logo 对应深色背景。")
        layout.addWidget(self._logo_row("方图 Logo", "1440 × 1440", "logo_square_dark", "logo_square_light"))
        layout.addWidget(self._logo_row("竖图 Logo", "1440 × 1920", "logo_tall_dark", "logo_tall_light"))
        rule = QLabel("上传要求：PNG、RGBA 透明通道、完整尺寸画布。")
        rule.setObjectName("settingsRule")
        layout.addWidget(rule)
        layout.addStretch()
        return page

    def _logo_row(self, title_text: str, ratio: str, dark_key: str, light_key: str) -> QFrame:
        row = QFrame()
        row.setObjectName("settingsRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 12, 0, 12)
        layout.setSpacing(28)
        label = QLabel(f"{title_text}\n{ratio}")
        label.setObjectName("settingsFieldLabel")
        label.setFixedWidth(140)
        layout.addWidget(label)
        layout.addWidget(self._logo_cell(dark_key, "浅色背景 · 深色 Logo", False), 1, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._logo_cell(light_key, "深色背景 · 浅色 Logo", True), 1, Qt.AlignmentFlag.AlignLeft)
        return row

    def _logo_cell(self, key: str, label_text: str, dark_background: bool) -> QWidget:
        cell = QWidget()
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        preview = QLabel()
        preview.setObjectName("logoPreviewDark" if dark_background else "logoPreviewLight")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Reserve one fixed preview column so square/tall variants align cleanly.
        preview_height = 86
        preview_width = preview_height if "square" in key else round(preview_height * 1440 / 1920)
        preview.setFixedSize(preview_width, preview_height)
        preview_holder = QWidget()
        preview_holder.setFixedSize(86, 86)
        preview_holder_layout = QHBoxLayout(preview_holder)
        preview_holder_layout.setContentsMargins(0, 0, 0, 0)
        preview_holder_layout.addWidget(preview, 0, Qt.AlignmentFlag.AlignCenter)
        path = self._current_logo_path(key)
        self.logo_paths[key] = path
        self.logo_previews[key] = preview
        self._update_preview(preview, path)
        layout.addWidget(preview_holder)
        controls = QVBoxLayout()
        controls.setSpacing(3)
        name = QLabel(label_text)
        name.setObjectName("logoVariantName")
        name.setWordWrap(True)
        controls.addWidget(name)
        filename = QLabel(path.name)
        filename.setObjectName("logoVariantMeta")
        filename.setWordWrap(True)
        controls.addWidget(filename)
        self.logo_filenames[key] = filename
        replace = QPushButton("更换")
        replace.setObjectName("logoReplaceButton")
        replace.clicked.connect(lambda _checked=False, logo_key=key: self._choose_logo(logo_key, preview, filename, label_text))
        controls.addWidget(replace, 0, Qt.AlignmentFlag.AlignLeft)
        controls.addStretch()
        layout.addLayout(controls, 1)
        return cell

    def _flat_about_page(self) -> QWidget:
        page, layout = self._flat_page(
            "关于 PixelFlow",
            "轻量、离线优先的电商主图整理工具，帮助你把原始素材整理成可直接检查的电商图片。",
        )
        for title_text, value_text in (
            ("这是什么", "PixelFlow 用于批量整理电商商品素材，按尺寸、品类模板和品牌 Logo 输出主图、透明产品图、唯品专享图及模特主图。"),
            ("怎么使用", "选择素材目录和输出目录 → 选择品类、尺寸及输出选项 → 必要时在素材确认中修正分类 → 点击生成，完成后可直接打开输出文件夹检查结果。"),
            ("素材目录", "支持图片直接放在根目录，也支持按颜色或款式放在子文件夹中；支持 JPG、JPEG、PNG。建议同一商品的图片放在同一文件夹，文件名尽量保留颜色或款式信息。"),
            ("使用技巧", "白底商品图适合生成普通主图；细节图可用于辅助判断；模特图请在文件名或文件夹名中标注“模特、上身、穿搭、model、lookbook”等关键词。Logo 请使用完整尺寸的 RGBA PNG，并按浅色背景/深色背景选择对应版本。文件命名建议使用简短模式，完整原名可在输出清单中追溯。"),
            ("功能边界", "模特图只做分类和安全构图放大，保留原图内容，不凭空生成新人物或新场景。本地 ONNX 模型只提供人物位置和构图建议，无法使用时自动回退规则模式；未启用远程辅助时无需 API Key，图片处理在本机完成。"),
        ):
            row = QFrame()
            row.setObjectName("settingsRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 11, 0, 11)
            row_layout.setSpacing(24)
            label = QLabel(title_text)
            label.setObjectName("settingsFieldLabel")
            label.setFixedWidth(82)
            value = QLabel(value_text)
            value.setObjectName("settingsAboutValue")
            value.setWordWrap(True)
            row_layout.addWidget(label)
            row_layout.addWidget(value, 1)
            layout.addWidget(row)
        layout.addStretch()
        return page

    def _output_options_page(self) -> QWidget:
        page, layout = self._flat_page("输出选项", "设置每个选项的默认状态，以及是否显示在主页。尺寸选择始终保留在主页。")
        header = QFrame()
        header.setObjectName("settingsRow")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 10, 0, 10)
        header_layout.setSpacing(16)
        header_layout.addWidget(QLabel("选项"), 1)
        default_label = QLabel("默认开启")
        default_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        default_label.setFixedWidth(78)
        header_layout.addWidget(default_label)
        home_label = QLabel("主页显示")
        home_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        home_label.setFixedWidth(78)
        header_layout.addWidget(home_label)
        layout.addWidget(header)
        self.output_defaults: dict[str, QCheckBox] = {}
        self.output_visibility: dict[str, QCheckBox] = {}
        for key, label_text, hint_text in self.OUTPUT_OPTIONS:
            row = QFrame()
            row.setObjectName("settingsRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 13, 0, 13)
            row_layout.setSpacing(16)
            text_column = QVBoxLayout()
            text_column.setSpacing(4)
            title = QLabel(label_text)
            title.setObjectName("settingsFieldLabel")
            text_column.addWidget(title)
            hint = QLabel(hint_text)
            hint.setObjectName("settingsFieldHint")
            text_column.addWidget(hint)
            row_layout.addLayout(text_column, 1)
            default_check = QCheckBox()
            default_check.setChecked(self.settings.value(f"output_default_{key}", key != "enable_material_understanding" or False, type=bool))
            default_check.setToolTip("生成任务开始时的默认状态")
            self.output_defaults[key] = default_check
            default_column = QWidget()
            default_column_layout = QHBoxLayout(default_column)
            default_column_layout.setContentsMargins(0, 0, 0, 0)
            default_column_layout.addStretch()
            default_column_layout.addWidget(default_check)
            default_column_layout.addStretch()
            default_column.setFixedWidth(78)
            row_layout.addWidget(default_column)
            visible_column = QWidget()
            visible_column_layout = QHBoxLayout(visible_column)
            visible_column_layout.setContentsMargins(0, 0, 0, 0)
            visible_column.setFixedWidth(78)
            if key != "include_model_images":
                visible_check = QCheckBox()
                visible_check.setChecked(self.settings.value(f"output_visible_{key}", True, type=bool))
                visible_check.setToolTip("是否显示在主页")
                self.output_visibility[key] = visible_check
                visible_column_layout.addStretch()
                visible_column_layout.addWidget(visible_check)
                visible_column_layout.addStretch()
            row_layout.addWidget(visible_column)
            layout.addWidget(row)
        naming_row = QFrame()
        naming_row.setObjectName("settingsRow")
        naming_layout = QHBoxLayout(naming_row)
        naming_layout.setContentsMargins(0, 13, 0, 13)
        naming_layout.setSpacing(16)
        naming_text = QVBoxLayout()
        naming_text.setSpacing(4)
        naming_title = QLabel("文件命名")
        naming_title.setObjectName("settingsFieldLabel")
        naming_text.addWidget(naming_title)
        naming_hint = QLabel("简短模式使用编号和颜色/款式名，原始完整路径会写入输出清单")
        naming_hint.setObjectName("settingsFieldHint")
        naming_hint.setWordWrap(True)
        naming_text.addWidget(naming_hint)
        naming_layout.addLayout(naming_text, 1)
        self.output_naming_mode = QComboBox()
        self.output_naming_mode.addItem("简短模式（推荐）", "short")
        self.output_naming_mode.addItem("保留原始名称", "original")
        saved_naming_mode = str(self.settings.value("output_naming_mode", "short") or "short")
        saved_index = self.output_naming_mode.findData(saved_naming_mode)
        self.output_naming_mode.setCurrentIndex(saved_index if saved_index >= 0 else 0)
        self.output_naming_mode.setMinimumWidth(150)
        naming_layout.addWidget(self.output_naming_mode)
        layout.addWidget(naming_row)
        layout.addStretch()
        return page

    def _flat_model_page(self) -> QWidget:
        page, layout = self._flat_page("模特图增强", "本地模型辅助判断人物位置、主体范围和安全构图，不生成新图片；不可用或低置信度时自动回退规则。")
        self.local_model_check = QCheckBox("启用本地模型辅助判断")
        self.local_model_check.setChecked(self.settings.value("local_model_enabled", False, type=bool))
        layout.addWidget(self.local_model_check)
        path_row = QHBoxLayout()
        self.local_model_path = QLineEdit(str(self.settings.value("local_model_path", "") or ""))
        self.local_model_path.setReadOnly(True)
        self.local_model_path.setPlaceholderText("尚未选择模型文件夹")
        path_row.addWidget(self.local_model_path, 1)
        choose = QPushButton("选择文件夹")
        choose.clicked.connect(self._choose_model_dir)
        path_row.addWidget(choose)
        layout.addLayout(path_row)
        self.local_model_status = QLabel()
        self.local_model_status.setObjectName("settingsFieldHint")
        layout.addWidget(self.local_model_status)
        test = QPushButton("测试模型")
        test.setObjectName("subtleButton")
        test.clicked.connect(self._test_model)
        layout.addWidget(test, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()
        self._refresh_model_status()
        return page

    def _save(self) -> None:
        for key, path in self.logo_paths.items():
            self.settings.setValue(f"{key}_path", str(path) if path != RESOURCE_ROOT / DEFAULT_LOGO_FILES[key] else "")
        for key, checkbox in self.output_defaults.items():
            self.settings.setValue(f"output_default_{key}", checkbox.isChecked())
        for key, checkbox in self.output_visibility.items():
            self.settings.setValue(f"output_visible_{key}", checkbox.isChecked())
        self.settings.setValue("output_visible_include_model_images", False)
        self.settings.setValue("output_naming_mode", self.output_naming_mode.currentData())
        self.settings.setValue("local_model_enabled", self.local_model_check.isChecked())
        self.settings.setValue("local_model_path", self.local_model_path.text().strip())
        self.accept()


class MainImageTool(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("window")
        self.setWindowTitle("PixelFlow")
        self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1120, 820)
        self.setMinimumSize(900, 740)
        self.product_root = QLineEdit()
        self.output_root = QLineEdit()
        self.size_checks = {name: QCheckBox(name) for name in SIZES}
        self.logo_check = QCheckBox("生成含 Logo / 无 Logo")
        self.vip_check = QCheckBox("生成唯品专享 1440")
        self.material_understanding_check = QCheckBox("启用素材理解层（规则识别）")
        self.model_images_check = QCheckBox("自动生成模特主图")
        self.home_option_cards: dict[str, QWidget] = {}
        self.category_template_choice = QComboBox()
        self.max_size_mb = QDoubleSpinBox()
        self.max_size_mb.setRange(0, 100)
        self.max_size_mb.setDecimals(2)
        self.max_size_mb.setSingleStep(0.1)
        self.max_size_mb.setSpecialValueText("不限制")
        self.max_size_mb.setSuffix(" MB")
        self.max_size_mb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.max_size_mb.setFixedWidth(150)
        self.start_button = QPushButton("开始生成")
        self.start_button.setEnabled(False)
        self.cancel_button = QPushButton("停止")
        self.cancel_button.setEnabled(False)
        self._job_running = False
        self.cancel_event = threading.Event()
        self.settings = QSettings("PixelFlow", "PixelFlow")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.status_phase = QLabel("准备就绪")
        self.status_phase.setObjectName("statusPhase")
        self.status = QLabel("准备就绪")
        self.status.setObjectName("statusText")
        self.status.setWordWrap(True)
        self._current_task_phase = "准备就绪"
        self._status_opacity = QGraphicsOpacityEffect(self.status_phase)
        self.status_phase.setGraphicsEffect(self._status_opacity)
        self._status_animation = QPropertyAnimation(self._status_opacity, b"opacity", self)
        self._status_animation.setDuration(160)
        self._status_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_events)
        self.timer.start(100)

    def _build_ui(self) -> None:
        """Keep the production surface aligned with the compact v1 workflow."""
        self._apply_theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 26, 48, 24)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(18)
        icon_label = QLabel()
        icon_label.setFixedSize(76, 76)
        icon_label.setPixmap(QIcon(str(APP_ICON_PATH)).pixmap(QSize(76, 76)))
        icon_label.setScaledContents(True)
        header.addWidget(icon_label)
        heading_host = QWidget()
        heading_host.setFixedHeight(60)
        heading = QVBoxLayout(heading_host)
        heading.setSpacing(2)
        heading.setContentsMargins(0, 0, 0, 0)
        title = QLabel("PixelFlow")
        title.setObjectName("title")
        title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        subtitle = QLabel("内置当前主图制作规范、Logo 和唯品会边框资源")
        subtitle.setObjectName("subtitle")
        subtitle.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addWidget(heading_host, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addStretch()
        settings_button = QPushButton("⚙")
        settings_button.setObjectName("iconButton")
        settings_button.setToolTip("设置")
        settings_button.setAccessibleName("设置")
        settings_button.setFixedSize(48, 48)
        settings_button.clicked.connect(self._open_settings)
        header.addWidget(settings_button, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(header)

        folders = QFrame()
        folders.setObjectName("panel")
        folders_layout = QHBoxLayout(folders)
        folders_layout.setContentsMargins(24, 20, 24, 20)
        folders_layout.setSpacing(24)
        for label_text, field, command in (
            ("产品素材文件夹", self.product_root, self._choose_product_root),
            ("输出文件夹", self.output_root, self._choose_output_root),
        ):
            column = QVBoxLayout()
            column.setSpacing(8)
            label = QLabel(label_text)
            label.setStyleSheet("font-size: 14px; font-weight: 600;")
            column.addWidget(label)
            column.addWidget(self._path_row(field, command))
            folders_layout.addLayout(column, 1)
        layout.addWidget(folders)

        options = QFrame()
        options.setObjectName("optionsPanel")
        options_layout = QVBoxLayout(options)
        options_layout.setContentsMargins(0, 4, 0, 0)
        options_layout.setSpacing(16)
        options_layout.addLayout(self._section_heading("输出选项"))

        size_layout = QHBoxLayout()
        size_layout.setSpacing(16)
        ratios = {"1440x1440": "1:1", "1440x1920": "3:4", "1125x1500": "3:4"}
        for name, check in self.size_checks.items():
            check.setText("")
            check.setFixedSize(22, 22)
            card = ToggleOptionCard(check)
            card.setObjectName("optionCard")
            card.setFixedHeight(84)
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(20, 12, 20, 12)
            card_layout.setSpacing(14)
            card_layout.addWidget(check, 0, Qt.AlignmentFlag.AlignVCenter)
            text_column = QVBoxLayout()
            text_column.setSpacing(4)
            size_title = QLabel(name)
            size_title.setStyleSheet("font-size: 16px; font-weight: 600;")
            text_column.addWidget(size_title)
            ratio = QLabel(ratios[name])
            ratio.setStyleSheet("font-size: 13px;")
            text_column.addWidget(ratio)
            card_layout.addLayout(text_column)
            check.stateChanged.connect(lambda state, option_card=card: self._sync_option_card(option_card, state))
            self._sync_option_card(card, check.checkState().value)
            size_layout.addWidget(card, 1)
        options_layout.addLayout(size_layout)

        secondary = QHBoxLayout()
        secondary.setSpacing(16)
        self._add_home_toggle(secondary, self.logo_check, "include_logo")
        self._add_home_toggle(secondary, self.vip_check, "include_vip")
        options_layout.addLayout(secondary)

        material_card = self._add_home_toggle(options_layout, self.material_understanding_check, "enable_material_understanding", "仅识别主商品图 / 细节图；品类模板由下方选项独立控制")

        category_layout = QHBoxLayout()
        category_label = QLabel("品类比例模板")
        category_layout.addWidget(category_label)
        category_layout.addStretch()
        self.category_template_choice.addItem("不启用（保持当前规则）", None)
        self.category_template_choice.addItem("自动识别（规则）", "auto")
        for label, category in (("上衣", "shirt"), ("长裤", "long_pants"), ("短裤", "shorts"), ("鞋", "shoes"), ("帽子", "hat"), ("袜子", "socks"), ("包", "bag"), ("配件", "accessories"), ("其他", "other")):
            self.category_template_choice.addItem(label, category)
        self.category_template_choice.setFixedWidth(190)
        category_layout.addWidget(self.category_template_choice)
        options_layout.addLayout(category_layout)

        limit_layout = QHBoxLayout()
        limit_label = QLabel("单张图片最大文件大小")
        limit_layout.addWidget(limit_label)
        limit_layout.addStretch()
        limit_layout.addWidget(self.max_size_mb)
        options_layout.addLayout(limit_layout)
        layout.addWidget(options)

        footer = QFrame()
        footer.setObjectName("footer")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(24, 18, 24, 18)
        footer_layout.setSpacing(18)
        status_column = QVBoxLayout()
        status_column.setSpacing(8)
        status_heading = QLabel("生成状态")
        status_heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        status_column.addWidget(status_heading)
        status_column.addWidget(self.status_phase)
        status_column.addWidget(self.status)
        footer_layout.addLayout(status_column, 1)
        self.cancel_button.setMinimumSize(96, 56)
        footer_layout.addWidget(self.cancel_button)
        self.start_button.setObjectName("mainStartButton")
        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.start_button.setIconSize(QSize(20, 20))
        self.start_button.setMinimumSize(190, 56)
        footer_layout.addWidget(self.start_button)
        layout.addWidget(footer)
        layout.addWidget(self.progress)
        self.start_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self._request_cancel)
        self.product_root.textChanged.connect(self._refresh_start_button)
        self.output_root.textChanged.connect(self._refresh_start_button)
        for check in self.size_checks.values():
            check.stateChanged.connect(self._refresh_start_button)
        self._restore_settings()

    def _add_home_toggle(self, parent_layout, checkbox: QCheckBox, key: str, hint: str = "") -> QFrame:
        card = ToggleOptionCard(checkbox)
        card.setObjectName("optionCard")
        card.setFixedHeight(58 if hint else 64)
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(20, 8, 20, 8)
        card_layout.addWidget(checkbox)
        if hint:
            card_layout.addStretch()
            description = QLabel(hint)
            description.setObjectName("settingHint")
            card_layout.addWidget(description)
        checkbox.stateChanged.connect(lambda state, option_card=card: self._sync_option_card(option_card, state))
        self._sync_option_card(card, checkbox.checkState().value)
        parent_layout.addWidget(card)
        self.home_option_cards[key] = card
        return card

    def _build_ui_legacy(self) -> None:
        self._apply_theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 26, 48, 24)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(18)
        icon_label = QLabel()
        icon_label.setFixedSize(76, 76)
        icon_label.setPixmap(QIcon(str(APP_ICON_PATH)).pixmap(QSize(76, 76)))
        icon_label.setScaledContents(True)
        header.addWidget(icon_label)
        heading = QVBoxLayout()
        heading.setSpacing(4)
        title = QLabel("PixelFlow")
        title.setObjectName("title")
        subtitle = QLabel("内置当前主图制作规范、Logo 和唯品会边框资源")
        subtitle.setObjectName("subtitle")
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading)
        header.addStretch()
        settings_button = QPushButton("⚙")
        settings_button.setObjectName("iconButton")
        settings_button.setToolTip("设置")
        settings_button.setAccessibleName("设置")
        settings_button.setFixedSize(64, 64)
        settings_button.clicked.connect(self._open_settings)
        header.addWidget(settings_button, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(header)

        folders = QFrame()
        folders.setObjectName("panel")
        folders_layout = QHBoxLayout(folders)
        folders_layout.setContentsMargins(24, 20, 24, 20)
        folders_layout.setSpacing(24)
        for label_text, field, command in (
            ("产品素材文件夹", self.product_root, self._choose_product_root),
            ("输出文件夹", self.output_root, self._choose_output_root),
        ):
            column = QVBoxLayout()
            column.setSpacing(8)
            label = QLabel(label_text)
            label.setStyleSheet("font-size: 14px; font-weight: 600;")
            column.addWidget(label)
            column.addWidget(self._path_row(field, command))
            folders_layout.addLayout(column, 1)
        layout.addWidget(folders)

        # Output choices can grow as features are enabled.  Keep the footer usable
        # on shorter displays instead of letting fixed-height cards overlap.
        options_scroll = QScrollArea()
        options_scroll.setObjectName("optionsScroll")
        options_scroll.setWidgetResizable(True)
        options_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        options_scroll.setFrameShape(QFrame.Shape.NoFrame)
        options = QFrame()
        options.setObjectName("optionsPanel")
        options_layout = QVBoxLayout(options)
        options_layout.setContentsMargins(0, 4, 0, 0)
        options_layout.setSpacing(16)
        options_layout.addLayout(self._section_heading("输出选项"))

        size_layout = QHBoxLayout()
        size_layout.setSpacing(16)
        ratios = {"1440x1440": "1:1", "1440x1920": "3:4", "1125x1500": "3:4"}
        for name, check in self.size_checks.items():
            check.setChecked(True)
            check.setText("")
            check.setFixedSize(22, 22)
            card = ToggleOptionCard(check)
            card.setObjectName("optionCard")
            card.setFixedHeight(84)
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(20, 12, 20, 12)
            card_layout.setSpacing(14)
            card_layout.addWidget(check, 0, Qt.AlignmentFlag.AlignVCenter)
            text_column = QVBoxLayout()
            text_column.setSpacing(4)
            title = QLabel(name)
            title.setStyleSheet("font-size: 16px; font-weight: 600;")
            text_column.addWidget(title)
            ratio = QLabel(ratios[name])
            ratio.setStyleSheet("font-size: 13px;")
            text_column.addWidget(ratio)
            card_layout.addLayout(text_column)
            check.stateChanged.connect(
                lambda state, option_card=card: self._sync_option_card(option_card, state)
            )
            self._sync_option_card(card, check.checkState().value)
            size_layout.addWidget(card, 1)
        options_layout.addLayout(size_layout)

        secondary = QHBoxLayout()
        secondary.setSpacing(16)
        self.logo_check.setChecked(True)
        self.vip_check.setChecked(True)
        for check in (self.logo_check, self.vip_check):
            card = ToggleOptionCard(check)
            card.setObjectName("optionCard")
            card.setFixedHeight(64)
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(20, 8, 20, 8)
            card_layout.addWidget(check)
            check.stateChanged.connect(
                lambda state, option_card=card: self._sync_option_card(option_card, state)
            )
            self._sync_option_card(card, check.checkState().value)
            secondary.addWidget(card, 1)
        options_layout.addLayout(secondary)

        material_card = ToggleOptionCard(self.material_understanding_check)
        material_card.setObjectName("optionCard")
        material_card.setFixedHeight(58)
        material_layout = QHBoxLayout(material_card)
        material_layout.setContentsMargins(20, 8, 20, 8)
        material_layout.addWidget(self.material_understanding_check)
        material_layout.addStretch()
        material_hint = QLabel("自动区分商品白底图、细节图与模特图；品类模板仅作用于商品白底图")
        material_hint.setStyleSheet("font-size: 13px;")
        material_layout.addWidget(material_hint)
        self.material_understanding_check.stateChanged.connect(
            lambda state: self._sync_option_card(material_card, state)
        )
        self._sync_option_card(material_card, self.material_understanding_check.checkState().value)
        options_layout.addWidget(material_card)

        self.model_images_check.setChecked(True)
        model_card = ToggleOptionCard(self.model_images_check)
        model_card.setObjectName("optionCard")
        model_card.setFixedHeight(64)
        model_layout = QHBoxLayout(model_card)
        model_layout.setContentsMargins(20, 8, 20, 8)
        model_layout.addWidget(self.model_images_check)
        model_layout.addStretch()
        model_hint = QLabel("保留模特原图背景，按安全构图输出")
        model_hint.setObjectName("settingHint")
        model_layout.addWidget(model_hint)
        self.model_images_check.stateChanged.connect(
            lambda state: self._sync_option_card(model_card, state)
        )
        self._sync_option_card(model_card, self.model_images_check.checkState().value)
        options_layout.addWidget(model_card)

        advanced_toggle = QPushButton("高级设置  ▼")
        advanced_toggle.setObjectName("advancedToggle")
        advanced_toggle.setCheckable(True)
        advanced_toggle.setChecked(True)
        options_layout.addWidget(advanced_toggle)
        advanced_panel = QFrame()
        advanced_panel.setObjectName("advancedPanel")
        advanced_panel.setVisible(True)
        advanced_layout = QVBoxLayout(advanced_panel)
        advanced_layout.setContentsMargins(12, 10, 12, 10)
        advanced_layout.setSpacing(12)
        category_layout = QHBoxLayout()
        category_label = QLabel("品类比例模板")
        category_layout.addWidget(category_label)
        category_layout.addStretch()
        self.category_template_choice.addItem("不启用（保持当前规则）", None)
        self.category_template_choice.addItem("自动识别（规则）", "auto")
        for label, category in (
            ("上衣", "shirt"),
            ("长裤", "long_pants"),
            ("短裤", "shorts"),
            ("鞋", "shoes"),
            ("帽子", "hat"),
            ("袜子", "socks"),
            ("包", "bag"),
            ("配件", "accessories"),
            ("其他", "other"),
        ):
            self.category_template_choice.addItem(label, category)
        self.category_template_choice.setFixedWidth(190)
        category_layout.addWidget(self.category_template_choice)
        advanced_layout.addLayout(category_layout)

        limit_layout = QHBoxLayout()
        limit_layout.setContentsMargins(0, 2, 0, 0)
        limit_label = QLabel("单张图片最大文件大小")
        limit_layout.addWidget(limit_label)
        limit_layout.addStretch()
        limit_layout.addWidget(self.max_size_mb)
        advanced_layout.addLayout(limit_layout)
        options_layout.addWidget(advanced_panel)
        advanced_toggle.toggled.connect(
            lambda checked: (
                advanced_panel.setVisible(checked),
                advanced_toggle.setText("高级设置  ▲" if checked else "高级设置  ▼"),
            )
        )
        advanced_toggle.setText("高级设置  ▲")
        options_scroll.setWidget(options)
        layout.addWidget(options_scroll, 1)

        footer = QFrame()
        footer.setObjectName("footer")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(24, 18, 24, 18)
        footer_layout.setSpacing(18)
        status_column = QVBoxLayout()
        status_column.setSpacing(8)
        status_heading = QLabel("生成状态")
        status_heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        status_column.addWidget(status_heading)
        status_column.addWidget(self.status)
        footer_layout.addLayout(status_column, 1)
        self.cancel_button.setMinimumSize(96, 56)
        footer_layout.addWidget(self.cancel_button)
        self.start_button.setObjectName("startButton")
        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.start_button.setIconSize(QSize(20, 20))
        self.start_button.setMinimumSize(190, 56)
        footer_layout.addWidget(self.start_button)
        layout.addWidget(footer)
        layout.addWidget(self.progress)
        self.start_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self._request_cancel)
        self.product_root.textChanged.connect(self._refresh_start_button)
        self.output_root.textChanged.connect(self._refresh_start_button)
        for check in self.size_checks.values():
            check.stateChanged.connect(self._refresh_start_button)
        self._restore_settings()

    def _open_settings(self) -> None:
        dialog = V15SettingsDialog(self, self.settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_output_preferences()

    def _apply_output_preferences(self) -> None:
        defaults = {
            "include_logo": True,
            "include_vip": True,
            "enable_material_understanding": False,
            "include_model_images": True,
        }
        controls = {
            "include_logo": self.logo_check,
            "include_vip": self.vip_check,
            "enable_material_understanding": self.material_understanding_check,
            "include_model_images": self.model_images_check,
        }
        for key, checkbox in controls.items():
            value = self.settings.value(f"output_default_{key}", defaults[key], type=bool)
            checkbox.setChecked(value)
        for key, card in self.home_option_cards.items():
            visible = (
                False
                if key == "include_model_images"
                else self.settings.value(f"output_visible_{key}", True, type=bool)
            )
            card.setVisible(visible)

    @staticmethod
    def _sync_option_card(card: QFrame, state: int) -> None:
        card.setProperty("selected", bool(state))
        style = card.style()
        style.unpolish(card)
        style.polish(card)
        card.update()

    @staticmethod
    def _system_uses_dark_mode() -> bool:
        window_color = QApplication.palette().color(QPalette.ColorRole.Window)
        return window_color.lightness() < 128

    def _apply_theme(self) -> None:
        dark = self._system_uses_dark_mode()
        if dark:
            colors = {
                "window": "#1b1f24",
                "text": "#eef2f7",
                "title": "#f7f9fc",
                "muted": "#aeb8c6",
                "panel": "#242a31",
                "border": "#3a444f",
                "card": "#242a31",
                "selected": "#243b52",
                "selected_border": "#78a9dc",
                "settings_selected": "#2d3f4f",
                "settings_selected_border": "#42576a",
                "settings_active_text": "#e8eef3",
                "settings_action": "#356b8e",
                "settings_action_hover": "#407b9f",
                "field": "#2a3038",
                "field_hover": "#313946",
                "field_border": "#4a5563",
                "button_hover": "#323d4b",
                "button_pressed": "#3b4d63",
                "button_hover_text": "#a9cdf2",
                "indicator": "#2a3038",
                "indicator_border": "#7f8da0",
                "progress": "#3a444f",
                "preview": "#20262d",
            }
        else:
            colors = {
                "window": "#f5f7fa",
                "text": "#20242b",
                "title": "#171a20",
                "muted": "#6f7783",
                "panel": "#ffffff",
                "border": "#e3e7ed",
                "card": "#ffffff",
                "selected": "#eef5fb",
                "selected_border": "#9fc5ef",
                "settings_selected": "#edf2f6",
                "settings_selected_border": "#c5d0da",
                "settings_active_text": "#315b7c",
                "settings_action": "#4d7ea7",
                "settings_action_hover": "#5c8db5",
                "field": "#fbfcfe",
                "field_hover": "#ffffff",
                "field_border": "#d9dfe7",
                "button_hover": "#f5f8fd",
                "button_pressed": "#eaf2ff",
                "button_hover_text": "#1d65d0",
                "indicator": "#ffffff",
                "indicator_border": "#b9c2cf",
                "progress": "#e4e9f0",
                "preview": "#f4f6f8",
            }
        self.setStyleSheet(
            f"""
            QWidget#window, QDialog {{ background: {colors['window']}; color: {colors['text']}; }}
            QLabel {{ color: {colors['text']}; }}
            QLabel#title {{ font-size: 30px; font-weight: 650; color: {colors['title']}; }}
            QLabel#subtitle {{ color: {colors['muted']}; font-size: 15px; }}
            QFrame#panel, QFrame#footer {{ background: {colors['panel']}; border: 1px solid {colors['border']}; border-radius: 14px; }}
            QFrame#optionsPanel {{ background: transparent; border: none; }}
            QScrollArea#optionsScroll {{ background: transparent; border: none; }}
            QFrame#settingsPanel, QFrame#advancedPanel {{ background: {colors['panel']}; border: 1px solid {colors['border']}; border-radius: 8px; }}
            QWidget#settingsRow {{ border-bottom: 1px solid {colors['border']}; }}
            QLabel#settingHint {{ color: {colors['muted']}; font-size: 12px; }}
            QLabel#statusPhase {{ color: #347ff0; font-size: 13px; font-weight: 650; }}
            QLabel#statusText {{ font-size: 14px; font-weight: 600; }}
            QFrame#optionCard {{ background: {colors['card']}; border: 1px solid {colors['border']}; border-radius: 12px; }}
            QFrame#optionCard[selected="true"] {{ background: {colors['selected']}; border: 1px solid {colors['selected_border']}; }}
            QLabel#preview {{ background: {colors['preview']}; border: 1px solid {colors['border']}; border-radius: 8px; }}
            QLineEdit, QDoubleSpinBox {{ background: {colors['field']}; border: 1px solid {colors['field_border']}; border-radius: 8px; padding: 10px 12px; color: {colors['text']}; selection-background-color: #2878e8; selection-color: #ffffff; }}
            QLineEdit:hover, QDoubleSpinBox:hover {{ border-color: #6b7f99; background: {colors['field_hover']}; }}
            QLineEdit:focus, QDoubleSpinBox:focus {{ border: 1px solid #2878e8; background: {colors['field_hover']}; }}
            QComboBox {{ min-height: 38px; background: {colors['field']}; border: 1px solid {colors['field_border']}; border-radius: 8px; padding: 0 10px; color: {colors['text']}; }}
            QComboBox:hover {{ border-color: #6b7f99; background: {colors['field_hover']}; }}
            QComboBox:focus {{ border: 1px solid #2878e8; background: {colors['field_hover']}; }}
            QComboBox QAbstractItemView {{ background: {colors['panel']}; color: {colors['text']}; selection-background-color: #2878e8; selection-color: #ffffff; border: 1px solid {colors['field_border']}; outline: none; }}
            QComboBox::drop-down {{ border: none; width: 26px; }}
            QComboBox#materialTypeChoice {{ min-height: 29px; max-height: 29px; border-radius: 9px; padding: 0 30px 0 10px; font-size: 12px; }}
            QComboBox#materialTypeChoice::drop-down {{ width: 28px; border: none; }}
            QComboBox#materialTypeChoice QAbstractItemView {{ border: 1px solid {colors['field_border']}; border-radius: 9px; padding: 4px; selection-background-color: {colors['selected']}; selection-color: {colors['text']}; }}
            QComboBox#materialTypeChoice QAbstractItemView::item {{ min-height: 27px; padding: 0 9px; border-radius: 6px; }}
            QComboBox#materialTypeChoice QAbstractItemView::item:selected {{ background: {colors['selected']}; color: {colors['text']}; }}
            QPushButton {{ background: {colors['card']}; border: 1px solid {colors['field_border']}; border-radius: 8px; padding: 10px 16px; color: {colors['text']}; }}
            QPushButton:hover {{ background: {colors['button_hover']}; border-color: #6b8fc7; color: {colors['button_hover_text']}; }}
            QPushButton:pressed {{ background: {colors['button_pressed']}; border-color: #2878e8; }}
            QPushButton#advancedToggle {{ background: transparent; border: none; padding: 6px 2px; text-align: left; font-weight: 600; }}
            QPushButton#advancedToggle:hover {{ color: #365f86; background: transparent; border: none; }}
            QPushButton#startButton {{ background: #287357; color: #ffffff; border: 1px solid #287357; border-radius: 10px; font-size: 16px; font-weight: 600; padding: 14px 26px; }}
            QPushButton#startButton:hover {{ background: #19372e; border-color: #19372e; color: #ffffff; }}
            QPushButton#startButton:pressed {{ background: #142b24; border-color: #142b24; }}
            QPushButton#startButton:disabled {{ background: #78a58f; border-color: #78a58f; color: #e5f2e9; }}
            QPushButton#mainStartButton {{ background: #347ff0; border-color: #347ff0; font-size: 16px; padding: 14px 26px; }}
            QPushButton#mainStartButton:hover {{ background: #216bdd; border-color: #216bdd; }}
            QPushButton#mainStartButton:pressed {{ background: #185dbf; border-color: #185dbf; }}
            QPushButton#mainStartButton:disabled {{ background: #6d91c5; border-color: #6d91c5; color: #dce8f8; }}
            QDialog QPushButton#startButton {{ background: {colors['settings_action']}; border-color: {colors['settings_action']}; font-size: 13px; padding: 9px 18px; border-radius: 8px; }}
            QDialog QPushButton#startButton:hover {{ background: {colors['settings_action_hover']}; border-color: {colors['settings_action_hover']}; }}
            QPushButton#iconButton {{ padding: 0; font-size: 28px; font-weight: 600; border-radius: 15px; }}
            QPushButton#iconButton:hover {{ background: #eef1f5; border-color: #d9e0e8; color: #4d5968; }}
            QPushButton#subtleButton {{ padding: 6px 10px; color: #365f86; background: transparent; border-color: transparent; }}
            QPushButton#subtleButton:hover {{ background: #eef5fb; border-color: #b9d2e8; }}
            QPushButton#settingsBackButton {{ padding: 0; color: {colors['muted']}; background: {colors['panel']}; border: 1px solid {colors['border']}; border-radius: 12px; font-size: 26px; }}
            QPushButton#settingsBackButton:hover {{ color: #365f86; background: #eef5fb; border-color: #9abfdf; }}
            QFrame#settingsTabs {{ background: {colors['panel']}; border: 1px solid {colors['border']}; border-radius: 10px; }}
            QScrollArea#settingsScroll {{ background: transparent; border: none; }}
            QPushButton#settingsTabButton {{ min-height: 38px; padding: 5px 8px; color: {colors['muted']}; background: transparent; border: 1px solid transparent; border-radius: 8px; font-size: 12px; font-weight: 600; }}
            QPushButton#settingsTabButton:hover {{ color: #365f86; background: {colors['settings_selected']}; }}
            QPushButton#settingsTabButton:checked {{ color: {colors['settings_active_text']}; background: {colors['settings_selected']}; border-color: {colors['settings_selected_border']}; }}
            QFrame#segmentedControl {{ background: {colors['field']}; border: 1px solid {colors['field_border']}; border-radius: 11px; }}
            QPushButton#segmentButton {{ min-height: 34px; padding: 5px 9px; color: {colors['muted']}; background: transparent; border: 1px solid transparent; border-radius: 8px; font-size: 12px; font-weight: 700; }}
            QPushButton#segmentButton:hover {{ color: #365f86; background: #eef5fb; }}
            QPushButton#segmentButton:checked {{ color: #ffffff; background: #2878e8; border-color: #2878e8; }}
            QWidget#settingsPage {{ background: {colors['window']}; }}
            QLabel#settingsEyebrow {{ color: #2878e8; font-size: 10px; font-weight: 800; letter-spacing: 1.4px; }}
            QLabel#settingsPageTitle {{ color: {colors['title']}; font-size: 22px; font-weight: 600; }}
            QLabel#settingsDescription {{ color: {colors['muted']}; font-size: 12px; line-height: 1.5; }}
            QFrame#settingsRow {{ background: transparent; border: none; border-bottom: 1px solid {colors['border']}; border-radius: 0; }}
            QLabel#settingsPage QLabel#settingsFieldLabel {{ font-size: 12px; font-weight: 600; }}
            QFrame#settingsCard {{ background: {colors['panel']}; border: 1px solid {colors['border']}; border-radius: 14px; }}
            QLabel#settingsCardTitle {{ color: {colors['text']}; font-size: 13px; font-weight: 600; }}
            QLabel#settingsCardHint, QLabel#settingsFieldHint {{ color: {colors['muted']}; font-size: 10px; }}
            QLabel#settingsFieldLabel {{ color: {colors['text']}; font-size: 11px; font-weight: 600; }}
            QFrame#logoAssetGroup {{ background: #f7faf7; border: 1px solid #dce5df; border-radius: 10px; }}
            QFrame#logoVariantLight, QFrame#logoVariantDark {{ border-radius: 8px; }}
            QFrame#logoVariantLight {{ background: #e8eeeb; border: 1px solid #d0dbd5; }}
            QFrame#logoVariantDark {{ background: #28433e; border: 1px solid #46635c; }}
            QLabel#logoPreviewLight {{ color: #30343a; background: #cbd0d5; border: 1px solid #aeb5bd; border-radius: 7px; font-size: 15px; font-weight: 800; letter-spacing: -1px; }}
            QLabel#logoPreviewDark {{ color: #f0f1f3; background: #3a3d43; border: 1px solid #555a62; border-radius: 7px; font-size: 15px; font-weight: 800; letter-spacing: -1px; }}
            QLabel#logoGroupTitle {{ color: {colors['text']}; font-size: 12px; font-weight: 800; }}
            QLabel#logoGroupMeta {{ color: {colors['muted']}; font-size: 10px; font-weight: 700; letter-spacing: .4px; }}
            QLabel#logoVariantName {{ color: {colors['text']}; font-size: 10px; font-weight: 600; }}
            QLabel#logoVariantMeta {{ color: {colors['muted']}; font-size: 9px; }}
            QFrame#logoVariantDark QLabel#logoVariantName {{ color: #f0f8f2; }}
            QFrame#logoVariantDark QLabel#logoVariantMeta {{ color: #a9c4b5; }}
            QPushButton#logoReplaceButton {{ min-height: 28px; padding: 4px 11px; color: {colors['text']}; background: {colors['card']}; border: 1px solid {colors['field_border']}; border-radius: 7px; font-size: 10px; font-weight: 600; }}
            QPushButton#logoReplaceButton:hover {{ color: {colors['settings_active_text']}; background: {colors['button_hover']}; border-color: {colors['settings_action']}; }}
            QPushButton#logoReplaceButton:pressed {{ background: {colors['button_pressed']}; border-color: {colors['settings_action_hover']}; }}
            QLabel#settingsRule {{ padding: 8px 0; color: {colors['muted']}; background: transparent; border: none; border-radius: 0; font-size: 10px; font-weight: 600; }}
            QLabel#settingsBadge {{ padding: 6px 9px; color: #287357; background: #e5f5ea; border-radius: 7px; font-size: 10px; font-weight: 600; }}
            QLabel#settingsAboutText, QLabel#settingsAboutValue {{ color: {colors['muted']}; font-size: 11px; line-height: 1.5; }}
            QCheckBox {{ spacing: 8px; font-size: 12px; color: {colors['text']}; }}
            QCheckBox::indicator {{ width: 17px; height: 17px; border: 1px solid {colors['indicator_border']}; border-radius: 5px; background: {colors['indicator']}; }}
            QCheckBox::indicator:hover {{ border-color: #2878e8; }}
            QCheckBox::indicator:checked {{ border-color: #2878e8; background: #2878e8; }}
            QSlider::groove:horizontal {{ height: 4px; background: {colors['progress']}; border-radius: 2px; }}
            QSlider::sub-page:horizontal {{ background: #2878e8; border-radius: 2px; }}
            QSlider::handle:horizontal {{ width: 14px; margin: -5px 0; border: 2px solid #2878e8; border-radius: 7px; background: #ffffff; }}
            QSlider::handle:horizontal:disabled {{ border-color: {colors['field_border']}; background: {colors['card']}; }}
            QProgressBar {{ background: {colors['progress']}; border: none; border-radius: 4px; height: 8px; }}
            QProgressBar::chunk {{ background: #4b9d78; border-radius: 4px; }}
            QToolTip {{ background: {colors['panel']}; color: {colors['text']}; border: 1px solid {colors['field_border']}; padding: 4px; }}
            """
        )

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.ApplicationPaletteChange:
            self._apply_theme()
        super().changeEvent(event)

    @staticmethod
    def _section_heading(text: str) -> QHBoxLayout:
        row = QHBoxLayout()
        bar = QFrame()
        bar.setFixedSize(5, 24)
        bar.setStyleSheet("background: #347ff0; border-radius: 2px;")
        row.addWidget(bar)
        label = QLabel(text)
        label.setStyleSheet("font-size: 18px; font-weight: 600;")
        row.addWidget(label)
        row.addStretch()
        return row

    def _path_row(self, field: QLineEdit, command) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)
        row_layout.addWidget(field)
        button = QPushButton("选择…")
        button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        button.clicked.connect(command)
        row_layout.addWidget(button)
        return row

    def _choose_product_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择产品素材文件夹")
        if path:
            product_root = Path(path)
            self.product_root.setText(path)
            output_name = product_root.name if product_root.name.endswith("主图") else f"{product_root.name}主图"
            self.output_root.setText(str(product_root.parent / output_name))

    def _choose_output_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_root.setText(path)

    @staticmethod
    def _has_material_sources(product_root: Path) -> bool:
        if not product_root.is_dir():
            return False
        try:
            entries = tuple(product_root.iterdir())
            return any(
                item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}
                for item in entries
            ) or any(
                child.is_dir()
                and not child.name.startswith(".")
                and any(
                    item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    for item in child.iterdir()
                )
                for child in entries
            )
        except OSError:
            return False

    def _refresh_start_button(self) -> None:
        product_root = Path(self.product_root.text().strip())
        selected_sizes = any(check.isChecked() for check in self.size_checks.values())
        can_start = (
            not self._job_running
            and bool(self.output_root.text().strip())
            and selected_sizes
            and self._has_material_sources(product_root)
        )
        self.start_button.setEnabled(can_start)

    def _restore_settings(self) -> None:
        self.settings.remove("product_root")
        self.settings.remove("output_root")
        self.product_root.clear()
        self.output_root.clear()
        self.logo_check.setChecked(self.settings.value("include_logo", True, type=bool))
        self.vip_check.setChecked(self.settings.value("include_vip", True, type=bool))
        self.material_understanding_check.setChecked(
            self.settings.value("enable_material_understanding", False, type=bool)
        )
        self._apply_output_preferences()
        self.max_size_mb.setValue(float(self.settings.value("max_size_mb", 0.0)))
        selected_sizes = self.settings.value("selected_sizes", list(SIZES))
        if isinstance(selected_sizes, str):
            selected_sizes = [selected_sizes]
        for name, check in self.size_checks.items():
            check.setChecked(name in selected_sizes)
        saved_category = self.settings.value("category_template", "") or None
        index = self.category_template_choice.findData(saved_category)
        if index >= 0:
            self.category_template_choice.setCurrentIndex(index)

    def _save_settings(self) -> None:
        self.settings.setValue("include_logo", self.logo_check.isChecked())
        self.settings.setValue("include_vip", self.vip_check.isChecked())
        self.settings.setValue("enable_material_understanding", self.material_understanding_check.isChecked())
        self.settings.setValue("include_model_images", self.model_images_check.isChecked())
        for key, checkbox in {
            "include_logo": self.logo_check,
            "include_vip": self.vip_check,
            "enable_material_understanding": self.material_understanding_check,
            "include_model_images": self.model_images_check,
        }.items():
            self.settings.setValue(f"output_default_{key}", checkbox.isChecked())
        self.settings.setValue("max_size_mb", self.max_size_mb.value())
        self.settings.setValue("selected_sizes", [name for name, check in self.size_checks.items() if check.isChecked()])
        self.settings.setValue("category_template", self.category_template_choice.currentData() or "")

    def _logo_overrides(self) -> dict[str, Path]:
        overrides: dict[str, Path] = {}
        key_mapping = {
            "logo_square_dark": "logo",
            "logo_square_light": "white_logo",
            "logo_tall_dark": "tall_logo",
            "logo_tall_light": "tall_white_logo",
        }
        for setting_key, engine_key in key_mapping.items():
            saved = str(self.settings.value(f"{setting_key}_path", "") or "")
            path = Path(saved)
            if path.is_file():
                overrides[engine_key] = path
        return overrides

    def _request_cancel(self) -> None:
        self.cancel_event.set()
        self.cancel_button.setEnabled(False)
        self._set_task_status("正在安全停止", "会完成当前图片，并保留已经生成的输出。")

    def _set_task_status(self, phase: str, detail: str) -> None:
        """Update the task state without making the worker thread touch the UI."""
        phase_changed = phase != self._current_task_phase
        self._current_task_phase = phase
        self.status_phase.setText(phase)
        self.status.setText(detail)
        if phase_changed:
            self._status_animation.stop()
            self._status_opacity.setOpacity(0.25)
            self._status_animation.setStartValue(0.25)
            self._status_animation.setEndValue(1.0)
            self._status_animation.start()

    @staticmethod
    def _progress_phase(message: str) -> str:
        if message.startswith(("素材识别：", "本地构图判断：", "AI卖点匹配：")):
            return "识别分类"
        if message.startswith(("透明图：", "唯品图：", "卖点图：")):
            return "整理输出"
        return "生成图片"

    def _open_output_folder(self, output_root: Path) -> None:
        if not output_root.is_dir():
            QMessageBox.warning(self, "输出文件夹不可用", f"未找到输出文件夹：\n{output_root}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_root)))

    def _show_failures_dialog(self, failures: list[dict[str, str]]) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"失败素材 · {len(failures)} 项")
        dialog.resize(680, 420)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(12)
        title = QLabel(f"失败素材（{len(failures)} 项）")
        title.setObjectName("settingsPageTitle")
        detail = QLabel("以下素材未能输出；其余成功结果不受影响。可根据文件名定位后重新生成。")
        detail.setObjectName("settingsDescription")
        detail.setWordWrap(True)
        entries = []
        for failure in failures:
            source = Path(str(failure.get("source", "未知素材")))
            message = str(failure.get("message", "未提供原因"))
            entries.append(f"{source.name}\n{message}")
        listing = QTextEdit(dialog)
        listing.setReadOnly(True)
        listing.setPlainText("\n\n".join(entries))
        close = QPushButton("关闭", dialog)
        close.setObjectName("startButton")
        close.clicked.connect(dialog.accept)
        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(close)
        layout.addWidget(title)
        layout.addWidget(detail)
        layout.addWidget(listing, 1)
        layout.addLayout(button_row)
        dialog.exec()

    def _show_generation_result(self, result: dict[str, object]) -> None:
        failures = [
            failure for failure in result.get("failures", [])
            if isinstance(failure, dict)
        ]
        output_root = Path(str(result.get("output_root", self.output_root.text().strip())))
        cancelled = bool(result.get("cancelled"))
        success_count = sum(
            int(result.get(key, 0) or 0)
            for key in ("main_images", "transparent_images", "vip_images", "selling_point_images")
        )
        skipped_count = int(result.get("skipped_model_sources", 0) or 0)
        dialog = QDialog(self)
        dialog.setWindowTitle("生成已停止" if cancelled else "生成完成")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(12)
        title = QLabel("已安全停止" if cancelled else "生成完成")
        title.setObjectName("settingsPageTitle")
        explanation = QLabel(
            "已保留已完成的输出；未开始的任务不会继续处理。"
            if cancelled
            else "输出已整理完成，可直接打开文件夹检查结果。"
        )
        explanation.setObjectName("settingsDescription")
        explanation.setWordWrap(True)
        summary = QLabel(
            f"成功输出：{success_count} 张\n"
            f"跳过素材：{skipped_count} 张\n"
            f"失败素材：{len(failures)} 张\n"
            f"自动并发：{int(result.get('render_workers', 1) or 1)}"
        )
        summary.setObjectName("settingsRule")
        summary.setWordWrap(True)
        location = QLabel(f"输出位置：{output_root}")
        location.setObjectName("settingsDescription")
        location.setWordWrap(True)
        open_button = QPushButton("打开输出文件夹", dialog)
        open_button.clicked.connect(lambda: self._open_output_folder(output_root))
        failure_button = QPushButton(f"查看失败素材（{len(failures)}）", dialog)
        failure_button.setVisible(bool(failures))
        failure_button.clicked.connect(lambda: self._show_failures_dialog(failures))
        close = QPushButton("完成", dialog)
        close.setObjectName("startButton")
        close.clicked.connect(dialog.accept)
        button_row = QHBoxLayout()
        button_row.addWidget(open_button)
        button_row.addWidget(failure_button)
        button_row.addStretch()
        button_row.addWidget(close)
        layout.addWidget(title)
        layout.addWidget(explanation)
        layout.addWidget(summary)
        layout.addWidget(location)
        layout.addLayout(button_row)
        dialog.exec()

    def _start(self) -> None:
        product_root = Path(self.product_root.text().strip())
        output_text = self.output_root.text().strip()
        selected_sizes = tuple(name for name, check in self.size_checks.items() if check.isChecked())
        if not product_root.is_dir():
            QMessageBox.warning(self, "缺少素材文件夹", "请选择有效的产品素材文件夹。")
            return
        if not self._has_material_sources(product_root):
            QMessageBox.warning(self, "没有图片素材", "文件夹内没有 JPG、JPEG 或 PNG 图片。")
            self._refresh_start_button()
            return
        if not output_text:
            QMessageBox.warning(self, "缺少输出文件夹", "请选择输出文件夹。")
            return
        if not selected_sizes:
            QMessageBox.warning(self, "缺少输出尺寸", "至少选择一个输出尺寸。")
            return
        self._save_settings()
        output_root = Path(output_text)
        include_logo = self.logo_check.isChecked()
        include_vip = self.vip_check.isChecked()
        include_model_images = self.model_images_check.isChecked()
        naming_mode = str(self.settings.value("output_naming_mode", "short") or "short")
        local_model_enabled = self.settings.value("local_model_enabled", False, type=bool)
        local_model_path = str(self.settings.value("local_model_path", "") or "")
        max_size_mb = self.max_size_mb.value() or None
        category_choice = self.category_template_choice.currentData()
        category_override = None if category_choice == "auto" else category_choice
        enable_category_template = category_choice is not None
        enable_material_understanding = (
            self.material_understanding_check.isChecked()
        )
        needs_material_classification = (
            enable_material_understanding or include_model_images
        )
        if needs_material_classification:
            self._job_running = True
            self.start_button.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setRange(0, max(1, len(self._material_sources(product_root))))
            self.progress.setValue(0)
            self._set_task_status("识别分类", "正在识别素材类型……")
        material_settings = (
            self._confirm_material_types(
                product_root,
                category_override,
                enable_category_template,
                local_model_enabled,
                local_model_path,
            )
            if needs_material_classification
            else ({}, {}, set())
        )
        if material_settings is None:
            self._job_running = False
            self.progress.setVisible(False)
            self._set_task_status("准备就绪", "已取消素材确认，可调整选项后重新开始。")
            self._refresh_start_button()
            return
        source_type_overrides, source_scale_adjustments, excluded_model_sources = material_settings
        logo_overrides = self._logo_overrides()
        output_root.mkdir(parents=True, exist_ok=True)
        self.cancel_event.clear()
        self._job_running = True
        self.start_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._set_task_status("准备素材", "正在整理生成任务……")
        threading.Thread(
            target=self._run,
            args=(
                product_root,
                output_root,
                selected_sizes,
                include_logo,
                include_vip,
                max_size_mb,
                category_override,
                enable_category_template,
                enable_material_understanding,
                source_type_overrides,
                source_scale_adjustments,
                excluded_model_sources,
                include_model_images,
                logo_overrides,
                naming_mode,
                local_model_enabled,
                local_model_path,
            ),
            daemon=True,
        ).start()

    def _run(
        self,
        product_root: Path,
        output_root: Path,
        selected_sizes: tuple[str, ...],
        include_logo: bool,
        include_vip: bool,
        max_size_mb: float | None,
        category_override: str | None,
        enable_category_template: bool,
        enable_material_understanding: bool,
        source_type_overrides: dict[str, str],
        source_scale_adjustments: dict[str, float],
        excluded_model_sources: set[str],
        include_model_images: bool,
        logo_overrides: dict[str, Path],
        naming_mode: str,
        local_model_enabled: bool,
        local_model_path: str,
    ) -> None:
        try:
            result = generate_images(
                product_root,
                output_root,
                resource_root=RESOURCE_ROOT,
                selected_sizes=selected_sizes,
                include_logo=include_logo,
                include_vip=include_vip,
                max_size_mb=max_size_mb,
                category_override=category_override,
                enable_category_template=enable_category_template,
                enable_material_understanding=enable_material_understanding,
                source_type_overrides=source_type_overrides,
                source_scale_adjustments=source_scale_adjustments,
                excluded_model_sources=excluded_model_sources,
                include_model_images=include_model_images,
                logo_overrides=logo_overrides,
                cancel_event=self.cancel_event,
                naming_mode=naming_mode,
                local_model_enabled=local_model_enabled,
                local_model_path=local_model_path,
                progress=lambda completed, total, message: self.events.put(
                    ("progress", (completed, total, message))
                ),
            )
            result["output_root"] = str(output_root)
            result["skipped_model_sources"] = len(excluded_model_sources)
            self.events.put(("done", result))
        except Exception as error:
            self.events.put(("error", error))

    @staticmethod
    def _material_sources(product_root: Path) -> tuple[Path, ...]:
        try:
            entries = tuple(sorted(product_root.iterdir()))
        except OSError:
            return ()
        material_dirs = []
        if any(
            entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for entry in entries
        ):
            material_dirs.append(product_root)
        material_dirs.extend(
            entry
            for entry in entries
            if entry.is_dir() and not entry.name.startswith(".")
        )
        return tuple(
            source
            for material_dir in material_dirs
            for source in sorted(material_dir.iterdir())
            if source.is_file() and source.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def _confirm_material_types(
        self,
        product_root: Path,
        category_override: str | None,
        enable_category_template: bool,
        local_model_enabled: bool,
        local_model_path: str,
    ) -> tuple[dict[str, str], dict[str, float], set[str]] | None:
        sources = self._material_sources(product_root)
        classifier = RuleSourceTypeClassifier()
        base_scale = None
        if enable_category_template:
            category = RuleCategoryClassifier().classify(
                product_root,
                manual_category=category_override,
            ).category
            base_scale = CategoryTemplateManager(
                RESOURCE_ROOT / "templates" / "category_templates.json"
            ).get(category, "1440x1440").scale

        def report_classification(completed, total, source, decision) -> None:
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(completed)
            self._set_task_status(
                "识别分类",
                f"{completed} / {total} · {source.name} → "
                f"{ {'main_product': '商品白底图', 'detail': '细节图', 'model': '模特图', 'model_detail': '模特图'}[decision.source_type] }"
            )
            QApplication.processEvents()

        decisions = classifier.classify_many(sources, progress=report_classification)
        # The confirmation UI intentionally has only three visible choices.
        # When the local pose model sees a clothing-only/partial model shot,
        # preserve it as ``model_detail`` internally but present it as the
        # existing "细节图" choice.  This keeps the review surface compact
        # while retaining the model-source behaviour (including Skip).
        if local_model_enabled:
            model_sources = [
                source for source in sources
                if decisions[source].source_type == "model"
            ]
            model_path = find_onnx_model(local_model_path)
            if model_sources and model_path is not None:
                try:
                    local_model = LocalModelAssistant(model_path)
                except Exception as error:  # noqa: BLE001 - preserve rule fallback
                    self._set_task_status("识别分类", f"本地模型不可用，已按规则继续：{error}")
                else:
                    self.progress.setRange(0, len(model_sources))
                    self.progress.setValue(0)
                    for index, source in enumerate(model_sources, start=1):
                        try:
                            composition = local_model.compose_for_sizes(source, SIZES)
                            if composition.get("model_view") == "detail":
                                decisions[source] = SourceTypeDecision(
                                    "model_detail",
                                    float(composition.get("confidence", 0.0)),
                                    "local_onnx",
                                    "本地模型识别为模特细节图，按细节图铺满处理",
                                    metadata=composition,
                                )
                        except Exception:
                            # A single photo failing inference must not block
                            # review or silently change its original decision.
                            pass
                        self.progress.setValue(index)
                        self._set_task_status(
                            "识别分类",
                            f"区分模特图与模特细节图 {index} / {len(model_sources)} · {source.name}",
                        )
                        QApplication.processEvents()
        dialog = QDialog(self)
        dialog.setWindowTitle(f"确认素材分类 · {len(sources)} 张")
        dialog.resize(1280, 780)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        summary = QLabel(dialog)
        summary.setObjectName("settingsDescription")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        grid_host = QWidget()
        grid_host.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred
        )
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(14)
        columns = 5
        type_choices: list[QComboBox] = []
        scale_sliders: list[QSlider] = []
        excluded_model_checks: list[QCheckBox] = []
        for row, source in enumerate(sources):
            card = QFrame(grid_host)
            card.setObjectName("optionCard")
            card.setFixedWidth(220)
            # Keep every material card the same height.  Otherwise a model
            # card becomes taller only when it shares a grid row with a
            # main-product card, which makes the skip control look cramped
            # or overly loose depending on its row.
            card.setFixedHeight(360)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(7)
            preview = QLabel(card)
            preview.setObjectName("preview")
            preview.setFixedSize(198, 138)
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            reader = QImageReader(str(source))
            source_size = reader.size()
            if source_size.isValid():
                scale = min(preview.width() / source_size.width(), preview.height() / source_size.height())
                reader.setScaledSize(
                    QSize(
                        max(1, round(source_size.width() * scale)),
                        max(1, round(source_size.height() * scale)),
                    )
                )
            pixmap = QPixmap.fromImage(reader.read())
            if pixmap.isNull():
                pixmap = QPixmap(str(source))
            if not pixmap.isNull():
                preview.setPixmap(
                    pixmap.scaled(
                        preview.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            card_layout.addWidget(preview)
            name_label = QLabel(source.name)
            name_label.setToolTip(source.relative_to(product_root).as_posix())
            name_label.setWordWrap(True)
            name_label.setMaximumHeight(36)
            name_label.setStyleSheet("font-size: 12px;")
            card_layout.addWidget(name_label)
            folder_label = QLabel(source.parent.name)
            folder_label.setStyleSheet("font-size: 11px;")
            folder_label.setWordWrap(True)
            folder_label.setMaximumHeight(24)
            card_layout.addWidget(folder_label)
            choice = QComboBox(card)
            choice.addItem("商品白底图", "main_product")
            choice.addItem(
                "细节图",
                "model_detail" if decisions[source].source_type == "model_detail" else "detail",
            )
            choice.addItem("模特图", "model")
            choice.setCurrentIndex(
                {"main_product": 0, "detail": 1, "model": 2, "model_detail": 1}[decisions[source].source_type]
            )
            choice.setObjectName("materialTypeChoice")
            choice.setToolTip(decisions[source].reason)
            card_layout.addWidget(choice)
            # Reserve the same breathing room above the model-only control
            # on every card, independent of the source name/folder length.
            card_layout.addSpacing(8)
            exclude_model_check = QCheckBox("不适用于本次品类（跳过）", card)
            exclude_model_check.setToolTip("仅作用于模特素材；勾选后，此图片不会生成到模特图输出。")
            card_layout.addWidget(exclude_model_check)
            scale_controls = QWidget(card)
            scale_layout = QVBoxLayout(scale_controls)
            scale_layout.setContentsMargins(0, 2, 0, 0)
            scale_layout.setSpacing(4)
            scale_row = QHBoxLayout()
            scale_row.setContentsMargins(0, 0, 0, 0)
            scale_row.setSpacing(6)
            scale_label = QLabel("主体大小 +0%")
            scale_label.setStyleSheet("font-size: 11px;")
            scale_row.addWidget(scale_label)
            scale_row.addStretch()
            reset_button = QPushButton("↺", scale_controls)
            reset_button.setFixedSize(24, 24)
            reset_button.setToolTip("恢复为 0%")
            reset_button.setStyleSheet("padding: 0; font-size: 15px;")
            scale_row.addWidget(reset_button)
            scale_layout.addLayout(scale_row)
            if base_scale is not None:
                base_label = QLabel(f"基础比例：{base_scale * 100:.1f}%")
                final_label = QLabel(f"最终比例：{base_scale * 100:.1f}%")
            else:
                base_label = QLabel("基础比例：当前规则")
                final_label = QLabel("最终比例：当前规则")
            base_label.setObjectName("settingHint")
            base_label.setStyleSheet("font-size: 11px;")
            final_label.setStyleSheet("font-size: 11px; font-weight: 600;")
            scale_layout.addWidget(base_label)
            scale_layout.addWidget(final_label)
            slider = ProductScaleSlider(Qt.Orientation.Horizontal, scale_controls)
            slider.setRange(-50, 50)
            slider.setValue(0)
            slider.setToolTip("仅作用于商品白底图；细节图和模特图不使用比例滑块")
            scale_layout.addWidget(slider)

            def sync_scale(
                value: int,
                label=scale_label,
                final=final_label,
                enabled_scale=base_scale,
            ) -> None:
                label.setText(f"主体大小 {value:+d}%")
                if enabled_scale is not None:
                    final.setText(
                        f"最终比例：{enabled_scale * (1 + value / 100) * 100:.1f}%"
                    )

            slider.valueChanged.connect(sync_scale)
            reset_button.clicked.connect(lambda _checked=False, control=slider: control.setValue(0))

            def sync_type_controls(
                _index: int,
                controls=scale_controls,
                exclude_check=exclude_model_check,
                current_choice=choice,
            ) -> None:
                is_main = current_choice.currentData() == "main_product"
                is_model = current_choice.currentData() in {"model", "model_detail"}
                controls.setVisible(is_main)
                exclude_check.setVisible(is_model)

            choice.currentIndexChanged.connect(sync_type_controls)
            sync_type_controls(choice.currentIndex())
            card_layout.addWidget(scale_controls)
            grid.addWidget(card, row // columns, row % columns)
            type_choices.append(choice)
            scale_sliders.append(slider)
            excluded_model_checks.append(exclude_model_check)
        grid_host.adjustSize()
        grid_host.setMinimumSize(grid.sizeHint())
        scroll.setWidget(grid_host)
        layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok,
            parent=dialog,
        )
        confirm_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        confirm_button.setText("确认并开始")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        def refresh_summary() -> None:
            counts = {"main_product": 0, "detail": 0, "model": 0}
            skipped = 0
            for choice, exclude_check in zip(type_choices, excluded_model_checks):
                source_type = str(choice.currentData())
                display_type = "detail" if source_type == "model_detail" else source_type
                if display_type in counts:
                    counts[display_type] += 1
                if source_type in {"model", "model_detail"} and exclude_check.isChecked():
                    skipped += 1
            process_count = len(sources) - skipped
            summary.setText(
                f"已识别 {len(sources)} 张 · 商品白底图 {counts['main_product']} · "
                f"细节图 {counts['detail']} · 模特图 {counts['model']} · 跳过 {skipped}"
            )
            confirm_button.setText(f"确认并开始（处理 {process_count} 张）")

        for choice, exclude_check in zip(type_choices, excluded_model_checks):
            choice.currentIndexChanged.connect(refresh_summary)
            exclude_check.toggled.connect(refresh_summary)
        refresh_summary()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return (
            {
                str(source): choice.currentData()
                for source, choice in zip(sources, type_choices)
                if (
                    choice.currentData() != decisions[source].source_type
                    or decisions[source].source_type == "model_detail"
                )
            },
            {
                str(source): float(slider.value())
                for source, choice, slider in zip(sources, type_choices, scale_sliders)
                if choice.currentData() == "main_product" and slider.value() != 0
            },
            {
                str(source)
                for source, choice, exclude_check in zip(sources, type_choices, excluded_model_checks)
                if choice.currentData() in {"model", "model_detail"} and exclude_check.isChecked()
            },
        )

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    completed, total, message = payload
                    self.progress.setRange(0, total)
                    self.progress.setValue(completed)
                    self._set_task_status(
                        self._progress_phase(str(message)),
                        f"{completed} / {total} · {message}",
                    )
                elif kind == "done":
                    result = payload
                    self.progress.setVisible(False)
                    self._job_running = False
                    self.cancel_button.setEnabled(False)
                    self._refresh_start_button()
                    self._set_task_status(
                        "已安全停止" if result.get("cancelled") else "已完成",
                        "已保留已完成的输出。" if result.get("cancelled") else "所有任务已完成，输出已整理。",
                    )
                    self._show_generation_result(result)
                elif kind == "error":
                    self.progress.setVisible(False)
                    self._job_running = False
                    self.cancel_button.setEnabled(False)
                    self._refresh_start_button()
                    self._set_task_status("生成失败", "任务未完成，请检查素材与输出位置后重试。")
                    QMessageBox.critical(self, "生成失败", str(payload))
        except queue.Empty:
            pass


if __name__ == "__main__":
    app = QApplication([])
    _install_crash_handler()
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = MainImageTool()
    window.show()
    app.exec()
