from __future__ import annotations

from dataclasses import dataclass
import unicodedata


DEFAULT_CATEGORIES = (
    "人名",
    "地名",
    "人物称呼",
    "第一人称",
    "口头禅",
    "学校制度",
    "住房环境",
)

LANGUAGE_OPTIONS = (
    "中文（简体）",
    "中文（繁体）",
    "日语",
    "英语",
    "韩语",
)

TRANSLATION_MODES = (
    "中文小说 → 日语本地化",
    "日语原文 → 改名润色",
)

FORCED_RENAME_CATEGORIES = frozenset(
    {"人名", "地名", "家族名", "组织名", "机构名", "种族名"}
)

AUTO_ENTITY_CATEGORIES = (
    "人名", "地名", "家族名", "组织名", "机构名", "种族名", "技能名", "道具名"
)


def is_valid_prompt_mapping(category: str, source: str, target: str) -> bool:
    return forced_rename_validation_error(category, source, target) is None


def forced_rename_validation_error(category: str, source: str, target: str) -> str | None:
    """Return a user-facing error when a forced rename is too close to its source."""
    source_value = source.strip()
    target_value = target.strip()
    if not source_value or not target_value:
        return "原词和替换词都不能为空。"
    if category not in FORCED_RENAME_CATEGORIES:
        return None
    source_compact = _compact_name(source_value)
    target_compact = _compact_name(target_value)
    if source_compact == target_compact:
        return f"“{source_value}”属于{category}，必须使用不同的新名称，且应有明显差异。"
    if source_compact in target_compact or target_compact in source_compact:
        return f"“{source_value}”的新名称与原名过于相近，请更换整体读音和结构。"
    if _edit_distance(source_compact, target_compact) <= max(1, min(len(source_compact), len(target_compact)) // 3):
        return f"“{source_value}”的新名称与原名过于相近，请更换整体读音和结构。"
    if _is_japanese_name(source_compact) and _is_japanese_name(target_compact):
        if _to_katakana(source_compact)[0] == _to_katakana(target_compact)[0]:
            return f"“{source_value}”的新名称不能沿用原名首音，请使用不同开头的名称。"
    return None


def _compact_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(char for char in normalized if char not in " ・·．.ー－-_/\\")


def _to_katakana(value: str) -> str:
    return "".join(chr(ord(char) + 0x60) if "ぁ" <= char <= "ゖ" else char for char in value)


def _is_japanese_name(value: str) -> bool:
    return bool(value) and all(
        "ァ" <= char <= "ヺ" or "ぁ" <= char <= "ゖ" for char in value
    )


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]

PROVIDER_DEFAULTS = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "",
    },
    "Anthropic": {
        "base_url": "https://api.anthropic.com",
        "model": "",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com",
        "model": "",
    },
    "OpenAI 兼容接口": {
        "base_url": "",
        "model": "",
    },
}


class ValidationError(ValueError):
    """用户输入或项目数据不合法。"""


@dataclass(frozen=True)
class ImportResult:
    imported: tuple[dict, ...]
    skipped: tuple[dict, ...]
