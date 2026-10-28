from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from novel_localizer import translator as translator_module
from novel_localizer.domain import ValidationError
from novel_localizer.storage import ProjectStore
from novel_localizer.translator import (
    ModelAPIError,
    ModelClient,
    ModelResponse,
    PromptBuilder,
    TranslationEngine,
    _remaining_forced_sources,
    apply_local_replacements,
    format_batch_start_log,
    normalize_replacement_category,
    parse_translation_json,
    split_source_text,
)


class FakeModelClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    def generate(
        self,
        profile,
        system_prompt,
        user_prompt,
        *,
        max_output_tokens,
        cancel_event=None,
    ):
        self.calls.append(
            {
                "profile": profile,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_output_tokens": max_output_tokens,
            }
        )
        return ModelResponse(
            text=json.dumps(self.payload, ensure_ascii=False),
            usage={"input_tokens": 100, "output_tokens": 80},
            response_id="test-response",
        )


class SequencedModelClient:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls = []

    def generate(
        self,
        profile,
        system_prompt,
        user_prompt,
        *,
        max_output_tokens,
        cancel_event=None,
    ):
        self.calls.append(
            {
                "profile": profile,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_output_tokens": max_output_tokens,
                "cancel_event": cancel_event,
            }
        )
        return ModelResponse(
            text=next(self.responses),
            usage={"input_tokens": 100, "output_tokens": 80},
            response_id="test-response",
        )


class LoggingSequencedModelClient(SequencedModelClient):
    def __init__(self, responses: list[str]):
        super().__init__(responses)
        self.logs = []

    def _log(self, message: str):
        self.logs.append(message)


class TruncatedThenSuccessClient(SequencedModelClient):
    def generate(self, *args, **kwargs):
        response = super().generate(*args, **kwargs)
        if len(self.calls) == 1:
            return ModelResponse(
                text=response.text,
                usage=response.usage,
                response_id=response.response_id,
                finish_reason="length",
            )
        return response


class AlwaysTruncatedClient:
    def __init__(self):
        self.calls = []

    def generate(
        self,
        profile,
        system_prompt,
        user_prompt,
        *,
        max_output_tokens,
        cancel_event=None,
    ):
        self.calls.append({
            "profile": profile,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "max_output_tokens": max_output_tokens,
        })
        return ModelResponse(
            text="",
            usage={"input_tokens": 100, "output_tokens": max_output_tokens},
            response_id="always-truncated",
            finish_reason="length",
        )


class FakeHTTPResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


class FakeHTTPSession:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeHTTPResponse(self.payload)


class SequencedHTTPSession:
    def __init__(self, responses: list[FakeHTTPResponse]):
        self.responses = iter(responses)
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append(
            {"url": url, "headers": headers, "json": dict(json), "timeout": timeout}
        )
        return next(self.responses)


class ModelClientTests(unittest.TestCase):
    def profile(self, provider: str, base_url: str) -> dict:
        return {
            "id": "p1",
            "provider": provider,
            "base_url": base_url,
            "model": "example-model",
            "api_key": "secret",
            "timeout_seconds": 90,
        }

    def test_openai_responses_request_and_parse(self):
        session = FakeHTTPSession(
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"translated_text":"译文"}'}
                        ],
                    }
                ],
                "usage": {"total_tokens": 10},
            }
        )
        result = ModelClient(session).generate(
            self.profile("OpenAI", "https://api.openai.com/v1"),
            "系统",
            "正文",
            max_output_tokens=2000,
        )
        self.assertEqual("https://api.openai.com/v1/responses", session.calls[0]["url"])
        self.assertEqual("系统", session.calls[0]["json"]["instructions"])
        self.assertIn("译文", result.text)

    def test_compatible_request_sends_extra_instructions_in_user_message(self):
        session = FakeHTTPSession(
            {
                "id": "chatcmpl_1",
                "choices": [{"message": {"content": '{"translated_text":"译文"}'}}],
                "usage": {},
            }
        )
        ModelClient(session).generate(
            self.profile("OpenAI 兼容接口", "https://relay.example/v1"),
            "系统",
            "用户附加要求：女主角使用敬语。\n\n【本章原文】\n正文",
            max_output_tokens=2000,
        )

        messages = session.calls[0]["json"]["messages"]
        self.assertEqual("user", messages[1]["role"])
        self.assertIn("女主角使用敬语", messages[1]["content"])

    def test_compatible_request_requests_json_object_mode(self):
        session = FakeHTTPSession(
            {
                "id": "chatcmpl_1",
                "choices": [{"message": {"content": '{"translated_text":"译文"}'}}],
                "usage": {},
            }
        )

        ModelClient(session).generate(
            self.profile("OpenAI 兼容接口", "https://relay.example/v1"),
            "系统",
            "正文",
            max_output_tokens=2000,
        )

        self.assertEqual(
            {"type": "json_object"}, session.calls[0]["json"]["response_format"]
        )

    def test_compatible_request_falls_back_when_json_mode_is_rejected(self):
        session = SequencedHTTPSession(
            [
                FakeHTTPResponse(
                    {"error": {"message": "response_format is unsupported"}},
                    status_code=400,
                ),
                FakeHTTPResponse(
                    {
                        "id": "chatcmpl_2",
                        "choices": [
                            {"message": {"content": '{"translated_text":"译文"}'}}
                        ],
                        "usage": {},
                    }
                ),
            ]
        )

        result = ModelClient(session).generate(
            self.profile("OpenAI 兼容接口", "https://relay.example/v1"),
            "系统",
            "正文",
            max_output_tokens=2000,
        )

        self.assertEqual("{\"translated_text\":\"译文\"}", result.text)
        self.assertIn("response_format", session.calls[0]["json"])
        self.assertNotIn("response_format", session.calls[1]["json"])

    def test_batch_log_reports_instruction_length_without_content(self):
        log = format_batch_start_log(
            self.profile("OpenAI 兼容接口", "https://relay.example/v1"),
            chapter_count=1,
            mode="翻译",
            max_output_tokens=2000,
            extra_instructions="不能写进日志的要求",
        )

        self.assertIn("本次额外要求=9 字", log)
        self.assertNotIn("不能写进日志的要求", log)

    def test_anthropic_messages_request_and_parse(self):
        session = FakeHTTPSession(
            {
                "id": "msg_1",
                "content": [{"type": "text", "text": "译文"}],
                "usage": {"input_tokens": 5},
            }
        )
        result = ModelClient(session).generate(
            self.profile("Anthropic", "https://api.anthropic.com"),
            "系统",
            "正文",
            max_output_tokens=2000,
        )
        self.assertEqual("https://api.anthropic.com/v1/messages", session.calls[0]["url"])
        self.assertEqual("2023-06-01", session.calls[0]["headers"]["anthropic-version"])
        self.assertEqual("译文", result.text)

    def test_gemini_interactions_request_and_parse(self):
        session = FakeHTTPSession(
            {
                "id": "interaction_1",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "译文"}],
                    }
                ],
                "usage": {"total_tokens": 8},
            }
        )
        result = ModelClient(session).generate(
            self.profile(
                "Google Gemini", "https://generativelanguage.googleapis.com"
            ),
            "系统",
            "正文",
            max_output_tokens=2000,
        )
        self.assertEqual(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            session.calls[0]["url"],
        )
        self.assertEqual("example-model", session.calls[0]["json"]["model"])
        self.assertFalse(session.calls[0]["json"]["store"])
        self.assertEqual("译文", result.text)

    def test_compatible_length_response_with_empty_content_is_preserved(self):
        session = FakeHTTPSession(
            {
                "id": "chatcmpl_empty_1",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "不能写入日志的内部内容",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 321,
                    "completion_tokens": 2000,
                    "total_tokens": 2321,
                },
            }
        )
        logs = []
        client = ModelClient(session, log_callback=logs.append)
        profile = self.profile(
            "OpenAI 兼容接口",
            "https://example.test/v1?api_key=do-not-log",
        )

        result = client.generate(
            profile,
            "不能写入日志的系统提示词",
            "不能写入日志的小说正文",
            max_output_tokens=2000,
        )

        self.assertEqual("", result.text)
        self.assertEqual("length", result.finish_reason)
        self.assertEqual("chatcmpl_empty_1", result.response_id)
        self.assertEqual(2000, result.usage["completion_tokens"])
        combined_logs = "\n".join(logs)
        self.assertIn("发送请求", combined_logs)
        self.assertIn("收到响应：HTTP 200", combined_logs)
        self.assertNotIn("do-not-log", combined_logs)
        self.assertNotIn("不能写入日志的系统提示词", combined_logs)
        self.assertNotIn("不能写入日志的小说正文", combined_logs)
        self.assertNotIn("不能写入日志的内部内容", combined_logs)

    def test_full_size_segment_caps_requested_output_tokens(self):
        self.assertEqual(
            12_000,
            translator_module._segment_output_token_limit(24_000, 4_500),
        )
        self.assertEqual(
            4_096,
            translator_module._segment_output_token_limit(24_000, 1_000),
        )
        self.assertEqual(
            1_000,
            translator_module._segment_output_token_limit(1_000, 4_500),
        )

    def test_short_segment_keeps_minimum_reasoning_budget(self):
        self.assertEqual(
            4_096,
            translator_module._segment_output_token_limit(24_000, 160),
        )
        self.assertEqual(
            4_096,
            translator_module._segment_output_token_limit(24_000, 550),
        )


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ProjectStore(self.root)
        self.chapter_path = self.root / "第一话.txt"
        self.chapter_path.write_text(
            "赵一在上海见到了钱二。", encoding="utf-8"
        )
        self.skill_path = self.root / "规则.md"
        self.skill_path.write_text(
            "医院名称也要本地化，并保持人物说话习惯。", encoding="utf-8"
        )
        self.project = self.store.create_project(
            "小说A",
            "中文（简体）",
            "日语",
            ["人名", "地名", "医院名"],
            [self.skill_path],
        )
        imported = self.store.import_chapters(
            self.project["id"], [self.chapter_path]
        )
        self.chapter = imported.imported[0]
        self.stored_source_path = (
            self.root
            / "projects"
            / self.project["id"]
            / "source"
            / self.chapter["stored_name"]
        )
        self.profile = {
            "id": "profile-1",
            "provider": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "model": "example-model",
            "api_key": "secret",
            "timeout_seconds": 120,
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_prompt_contains_existing_mapping_and_skill(self):
        self.store.upsert_chapter_mapping(
            self.project["id"], self.chapter["id"], "人名", "赵一", "田村 花子"
        )
        _, prompt, source = PromptBuilder(self.store).build(
            self.project["id"],
            self.chapter["id"],
            mode="翻译",
        )
        self.assertIn("田村 花子", prompt)
        self.assertIn("医院名称也要本地化", prompt)
        self.assertIn("赵一在上海", source)

    def test_prompt_excludes_mappings_absent_from_segment_and_context(self):
        self.store.upsert_chapter_mapping(
            self.project["id"], self.chapter["id"], "人名", "赵一", "田村花子"
        )
        self.store.upsert_chapter_mapping(
            self.project["id"], self.chapter["id"], "人名", "钱二", "铃木玲子"
        )

        _, prompt, _ = PromptBuilder(self.store).build(
            self.project["id"], self.chapter["id"], mode="翻译",
            source_override="赵一走进房间。",
        )

        self.assertIn('"source":"赵一"', prompt)
        self.assertNotIn('"source":"钱二"', prompt)

    def test_prompt_keeps_temporary_mapping_used_by_continuity_context(self):
        _, prompt, _ = PromptBuilder(self.store).build(
            self.project["id"], self.chapter["id"], mode="翻译",
            source_override="她笑了。",
            previous_source_tail="赵一站了起来。",
            temporary_replacements={
                "人名": [{"source": "赵一", "target": "田村花子", "notes": ""}]
            },
        )

        self.assertIn('"source":"赵一"', prompt)

    def test_translation_logs_relevant_term_context_size_without_term_content(self):
        client = LoggingSequencedModelClient([
            json.dumps({
                "translated_text": "田村花子は東京にいる。",
                "new_replacements": [], "warnings": [],
            }, ensure_ascii=False)
        ])

        TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile
        )

        logs = "\n".join(client.logs)
        self.assertIn("术语上下文：", logs)
        self.assertNotIn("赵一在上海见到了钱二。", logs)

    def test_disabled_rename_project_preserves_proper_names(self):
        self.store.upsert_replacement(
            self.project["id"], "人名", "赵一", "田村 花子"
        )
        self.store.update_project_settings(
            self.project["id"], proper_noun_localization=False
        )

        system_prompt, prompt, _ = PromptBuilder(self.store).build(
            self.project["id"], self.chapter["id"], mode="润色"
        )

        self.assertNotIn("田村 花子", prompt)
        self.assertNotIn("必须彻底改名", system_prompt)
        self.assertIn("保留原有专名", system_prompt)

    def test_review_polish_rewrites_the_full_chapter_without_global_name_chaining(self):
        original_source = self.store.read_chapter_text(
            self.project["id"], self.chapter["id"]
        )
        self.store.save_translation(
            self.project["id"], self.chapter["id"], "凪和悠斗见面。",
            profile_id="p", provider="test", model="m",
            review_status="需要复核", residual_sources=["凪", "悠斗"],
        )
        self.store.upsert_replacement(
            self.project["id"], "人名", "凪", "悠斗"
        )
        self.store.upsert_replacement(
            self.project["id"], "人名", "悠斗", "佐原蓮"
        )
        client = FakeModelClient({
            "translated_text": "白石千岁和佐原莲见面。",
            "name_decisions": [
                {"source": "凪", "decision": "substring", "reason": "仅为姓氏的一部分"},
                {"source": "悠斗", "decision": "renamed", "target": "佐原莲"},
            ],
            "remaining_actual_entities": [],
            "warnings": [],
        })

        result = TranslationEngine(self.store, client).polish_review_chapter(
            self.project["id"], self.chapter["id"], self.profile,
            max_output_tokens=16000,
        )

        saved = self.store.get_chapter(self.project["id"], self.chapter["id"])
        self.assertEqual("白石千岁和佐原莲见面。", result["translated_text"])
        self.assertEqual("完成", result["review_status"])
        self.assertEqual("已翻译", saved["status"])
        self.assertEqual([], saved["translations"][-1]["residual_sources"])
        self.assertEqual(original_source, self.store.read_chapter_text(
            self.project["id"], self.chapter["id"]
        ))
        self.assertIn("完整原文", client.calls[0]["user_prompt"])
        self.assertIn("旧译文", client.calls[0]["user_prompt"])

    def test_disabled_rename_project_bypasses_forced_mapping_pipeline(self):
        self.store.upsert_replacement(
            self.project["id"], "人名", "赵一", "田村 花子"
        )
        self.store.update_project_settings(
            self.project["id"], proper_noun_localization=False
        )
        client = FakeModelClient({
            "translated_text": "赵一在上海见到了钱二。",
            "new_replacements": [
                {
                    "category": "人名",
                    "source": "钱二",
                    "target": "铃木玲子",
                    "notes": "模型自行改名",
                }
            ],
            "warnings": [],
        })

        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile, mode="润色"
        )

        self.assertEqual("赵一在上海见到了钱二。", result["translated_text"])
        self.assertEqual(0, result["new_replacements"])
        self.assertEqual([], result["residual_sources"])
        self.assertNotIn(
            "钱二",
            self.store.load_registry(self.project["id"], "人名")["entries"],
        )

    def test_disabled_rename_project_does_not_confirm_prepared_name_candidates(self):
        self.store.update_project_settings(
            self.project["id"], proper_noun_localization=False
        )
        client = FakeModelClient({
            "entities": [
                {
                    "category": "人名",
                    "source": "赵一",
                    "target": "田村花子",
                    "notes": "模型自行改名",
                }
            ],
            "chapter_memory": {
                "summary": "赵一到了上海。",
                "character_states": [],
                "unresolved_threads": [],
            },
            "warnings": [],
        })

        result = TranslationEngine(self.store, client).prepare_entities(
            self.project["id"], [self.chapter["id"]], self.profile, mode="润色"
        )

        self.assertIn("保留原有专名", client.calls[0]["user_prompt"])
        self.assertEqual([], result["entities"])
        self.assertEqual(
            {}, self.store.load_registry(self.project["id"], "人名")["entries"]
        )

    def test_disabled_rename_project_blocks_traditional_category_alias(self):
        self.store.ensure_categories(self.project["id"], ["種族名"])
        self.store.upsert_replacement(
            self.project["id"], "種族名", "赵一", "妖精"
        )
        self.store.update_project_settings(
            self.project["id"], proper_noun_localization=False
        )

        _, prompt, _ = PromptBuilder(self.store).build(
            self.project["id"], self.chapter["id"], mode="润色"
        )
        self.assertNotIn("妖精", prompt)

        client = FakeModelClient({
            "entities": [
                {
                    "category": "種族名",
                    "source": "钱二",
                    "target": "兽人",
                    "notes": "模型自行改名",
                }
            ],
            "chapter_memory": {},
            "warnings": [],
        })
        result = TranslationEngine(self.store, client).prepare_entities(
            self.project["id"], [self.chapter["id"]], self.profile, mode="润色"
        )

        self.assertEqual([], result["entities"])
        self.assertNotIn(
            "钱二",
            self.store.load_registry(self.project["id"], "種族名")["entries"],
        )

    def test_pending_file_skill_is_analyzed_from_its_saved_content(self):
        client = FakeModelClient({"replacement_rules": [], "style_rules": []})

        analyzed = TranslationEngine(self.store, client).analyze_pending_skills(
            self.project["id"], self.profile
        )

        self.assertEqual(1, len(analyzed))
        self.assertIn("医院名称也要本地化", client.calls[0]["user_prompt"])
        self.assertEqual(
            self.project["skills"][0]["sha256"],
            self.store.load_project(self.project["id"])["skills"][0]["analysis"]["source_hash"],
        )

    def test_modes_are_explicit_and_unknown_mode_is_rejected(self):
        builder = PromptBuilder(self.store)
        _, localization_prompt, _ = builder.build(
            self.project["id"], self.chapter["id"],
            mode="翻译",
        )
        _, rewrite_prompt, _ = builder.build(
            self.project["id"], self.chapter["id"],
            mode="润色",
        )

        self.assertIn("将原文语言转换为目标语言", localization_prompt)
        self.assertIn("以目标语言输出", rewrite_prompt)
        with self.assertRaisesRegex(ValidationError, "处理模式"):
            builder.build(
                self.project["id"], self.chapter["id"], mode="本地化改写"
            )

    def test_prompt_ignores_historical_identity_name_mapping(self):
        registry_path = (
            self.store.project_dir(self.project["id"])
            / "replacements" / "人名.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry["entries"]["赵一"] = {
            "source": "赵一", "target": "赵一", "status": "已确认",
            "notes": "历史错误", "source_chapters": [self.chapter["id"]],
        }
        registry_path.write_text(
            json.dumps(registry, ensure_ascii=False), encoding="utf-8"
        )

        _, prompt, _ = PromptBuilder(self.store).build(
            self.project["id"], self.chapter["id"],
            mode="翻译",
        )

        self.assertNotIn('"source":"赵一","target":"赵一"', prompt)

    def test_prepare_entities_includes_complete_skill_and_selected_mode(self):
        client = FakeModelClient(
            {
                "entities": [
                    {"category": "人名", "source": "赵一", "target": "田村花子", "notes": "主角"}
                ],
                "chapter_memory": {"summary": "赵一到了上海。", "character_states": [], "unresolved_threads": []},
                "warnings": [],
            }
        )

        result = TranslationEngine(self.store, client).prepare_entities(
            self.project["id"], [self.chapter["id"]], self.profile,
            mode="翻译",
        )

        prompt = client.calls[0]["user_prompt"]
        self.assertIn("医院名称也要本地化", prompt)
        self.assertIn("翻译", prompt)
        self.assertEqual("已确认", result["entities"][0]["status"])
        self.assertEqual("田村花子", result["entities"][0]["target"])

    def test_prepare_entities_prioritizes_api_review_sources(self):
        client = FakeModelClient({
            "entities": [], "chapter_memory": {}, "warnings": [],
        })

        TranslationEngine(self.store, client).prepare_entities(
            self.project["id"], [self.chapter["id"]], self.profile,
            mode="翻译", focus_sources=["司"],
        )

        prompt = client.calls[0]["user_prompt"]
        self.assertIn("必须核实的残留词", prompt)
        self.assertIn("司", prompt)

    def test_translation_starts_without_separate_entity_preparation(self):
        client = FakeModelClient(
            {"translated_text": "田村花子は東京にいる。", "new_replacements": [], "warnings": []}
        )
        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile,
            mode="翻译", allow_unprepared=False,
        )
        self.assertEqual("田村花子は東京にいる。", result["translated_text"])
        self.assertEqual(1, len(client.calls))

    def test_direct_translation_accepts_new_family_name_without_preparation(self):
        self.chapter_path.write_text("アルハイム家の令嬢。", encoding="utf-8")
        client = FakeModelClient(
            {
                "translated_text": "ベルシュタイン家の令嬢。",
                "new_replacements": [
                    {"category": "家族名", "source": "アルハイム", "target": "ベルシュタイン", "notes": "家名"}
                ],
                "warnings": [],
            }
        )
        # Imported chapter content is immutable, so create a dedicated chapter.
        family_source = self.root / "家名.txt"
        family_source.write_text("アルハイム家の令嬢。", encoding="utf-8")
        chapter = self.store.import_chapters(
            self.project["id"], [family_source]
        ).imported[0]

        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], chapter["id"], self.profile,
            mode="润色",
        )

        self.assertEqual(1, result["new_replacements"])
        mappings = self.store.chapter_mappings(
            self.project["id"], chapter["id"]
        )
        registry = self.store.load_registry(self.project["id"], "家族名")
        self.assertEqual("ベルシュタイン", mappings["家族名"]["アルハイム"]["target"])
        self.assertNotIn("アルハイム", registry["entries"])

    def test_residual_forced_name_requires_review_without_automatic_retry(self):
        self.store.upsert_replacement(
            self.project["id"], "人名", "赵一", "田村花子"
        )
        client = FakeModelClient(
            {"translated_text": "赵一は東京にいる。", "new_replacements": [], "warnings": []}
        )
        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile,
            mode="翻译", allow_unprepared=True,
        )
        self.assertEqual("完成", result["review_status"])
        self.assertEqual([], result["residual_sources"])
        self.assertFalse(result["can_retry"])
        self.assertEqual(1, len(client.calls))
        self.assertEqual("翻译", result["record"]["mode"])
        self.assertEqual("完成", result["record"]["review_status"])

    def test_katakana_name_does_not_match_inside_a_longer_name(self):
        self.store.confirm_replacement(
            self.project["id"], "人名", "リーナ", "ミレイ", allow_user_override=True
        )
        replacements = self.store.load_all_replacements(self.project["id"])

        self.assertEqual("カトリーナが来た。", apply_local_replacements(
            "カトリーナが来た。", replacements
        ))
        self.assertEqual([], _remaining_forced_sources("カトリーナが来た。", replacements))

    def test_refresh_residual_review_clears_historical_katakana_substring_false_positive(self):
        self.store.confirm_replacement(
            self.project["id"], "人名", "リーナ", "ミレイ", allow_user_override=True
        )
        self.store.save_translation(
            self.project["id"], self.chapter["id"], "カトリーナが来た。",
            profile_id="", provider="", model="", usage={}, replacements_added=0,
            warnings=["旧版误报"], mode="润色",
            review_status="需要复核", residual_sources=["リーナ"],
            memory_version=0, skill_hash="", skill_files=[],
        )

        refreshed = TranslationEngine(self.store).refresh_residual_review(
            self.project["id"], self.chapter["id"]
        )

        self.assertEqual("完成", refreshed["review_status"])
        self.assertEqual([], refreshed["residual_sources"])

    def test_normalizes_japanese_race_category_to_project_category(self):
        self.assertEqual(
            "种族名",
            normalize_replacement_category("種族名", {"人名", "种族名"}),
        )

    def test_manual_residual_mapping_saves_a_new_localized_version(self):
        client = FakeModelClient(
            {"translated_text": "暗黒騎士が現れた。", "new_replacements": [], "warnings": []}
        )
        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile,
            mode="润色", allow_unprepared=True,
        )
        self.assertIn("暗黒騎士", result["translated_text"])

        reviewed = TranslationEngine(self.store).review_residuals(
            self.project["id"], self.chapter["id"], [{
                "action": "map", "category": "种族名",
                "source": "暗黒騎士", "target": "冥界骑士",
            }],
        )

        self.assertEqual("冥界骑士が現れた。", reviewed["translated_text"])
        self.assertEqual([], reviewed["residual_sources"])
        self.assertEqual("完成", reviewed["record"]["review_status"])

    def test_prompt_includes_confirmed_project_memory(self):
        from novel_localizer.project_memory import ProjectMemoryStore
        memory = ProjectMemoryStore(self.store)
        memory.save_candidates(
            self.project["id"], [self.chapter["id"]],
            [{
                "category": "人名", "source": "赵一", "target": "田村花子",
                "notes": "主角", "details": {"voice": "冷静"},
            }],
            chapter_memory={
                "summary": "赵一抵达上海。",
                "character_states": ["赵一正在车站"],
                "unresolved_threads": ["钱二为何出现"],
            },
        )
        memory.confirm_candidates(
            self.project["id"],
            [{
                "category": "人名", "source": "赵一", "target": "田村花子",
                "notes": "主角", "details": {"voice": "冷静"},
            }],
        )

        _, prompt, _ = PromptBuilder(self.store).build(
            self.project["id"], self.chapter["id"],
            mode="翻译",
        )
        self.assertIn("项目长期记忆", prompt)
        self.assertIn("赵一抵达上海", prompt)
        self.assertIn("钱二为何出现", prompt)

    def test_split_source_text_preserves_content_and_limits_chunks(self):
        source = "甲乙丙\n\n丁戊己\n庚辛壬\n\n" + "癸" * 13

        chunks = split_source_text(source, max_chars=10)

        self.assertEqual(source, "".join(chunks))
        self.assertTrue(all(0 < len(chunk) <= 10 for chunk in chunks))
        self.assertEqual(["短正文"], split_source_text("短正文", max_chars=10))

    def test_split_source_segments_marks_natural_and_hard_boundaries(self):
        splitter = getattr(
            translator_module,
            "split_source_segments",
            lambda source_text, max_chars: [],
        )
        paragraph = splitter(
            "甲乙。\n\n丙丁。", max_chars=5
        )
        self.assertEqual(
            ["start", "paragraph"],
            [item.boundary_before for item in paragraph],
        )
        self.assertEqual("甲乙。\n\n丙丁。", "".join(item.text for item in paragraph))

        sentence = splitter(
            "甲乙。丙丁。", max_chars=4
        )
        self.assertEqual(
            ["start", "sentence"],
            [item.boundary_before for item in sentence],
        )

        hard = splitter(
            "甲乙丙丁戊己", max_chars=3
        )
        self.assertEqual(
            ["start", "hard"],
            [item.boundary_before for item in hard],
        )

    def test_split_source_segments_does_not_treat_partial_crlf_as_natural(self):
        segments = translator_module.split_source_segments(
            "甲乙\r\n丙丁", max_chars=3
        )

        self.assertEqual("甲乙\r\n丙丁", "".join(item.text for item in segments))
        self.assertEqual("hard", segments[1].boundary_before)

        complete_crlf = translator_module.split_source_segments(
            "甲\r\n乙", max_chars=3
        )
        self.assertEqual("paragraph", complete_crlf[1].boundary_before)

    def test_context_helpers_are_semantic_and_bounded(self):
        missing = lambda text, max_chars: ""
        source_context = getattr(
            translator_module, "_localization_source_context", missing
        )
        translation_context = getattr(
            translator_module, "_translation_sentence_context", missing
        )
        hard_context = getattr(
            translator_module, "_hard_cut_source_context", missing
        )

        source = "前の段落。\n\n最後の段落一。最後の段落二。"
        self.assertEqual(
            "最後の段落一。最後の段落二。",
            source_context(source, max_chars=50),
        )
        self.assertEqual(
            "最後の文。",
            translation_context("古い文。最後の文。", max_chars=6),
        )
        self.assertEqual("丁戊己", hard_context("甲乙丙丁戊己", max_chars=3))
        self.assertLessEqual(len(source_context("長" * 900, 500)), 500)

    def test_localization_source_context_uses_last_single_newline_paragraph(self):
        context = translator_module._localization_source_context(
            "第一段。\n第二段。\n", max_chars=500
        )
        crlf_context = translator_module._localization_source_context(
            "第一段。\r\n第二段。\r\n", max_chars=500
        )

        self.assertEqual("第二段。", context)
        self.assertEqual("第二段。", crlf_context)

    def test_segment_prompt_includes_read_only_context_and_temporary_terms(self):
        _, prompt, source = PromptBuilder(self.store).build(
            self.project["id"],
            self.chapter["id"],
            mode="翻译",
            source_override="钱二继续向前走。",
            segment_index=2,
            segment_count=4,
            previous_source_tail="赵一叫住了钱二。",
            previous_translation_tail="田村花子は鈴木玲子を呼び止めた。",
            temporary_replacements={
                "人名": [
                    {
                        "source": "钱二",
                        "target": "鈴木玲子",
                        "notes": "本章前段首次出现",
                    }
                ]
            },
        )

        self.assertEqual("钱二继续向前走。", source)
        self.assertIn("第 2/4 段", prompt)
        self.assertIn("只读上下文", prompt)
        self.assertIn("不得重复翻译", prompt)
        self.assertIn("赵一叫住了钱二。", prompt)
        self.assertIn("田村花子は鈴木玲子を呼び止めた。", prompt)
        self.assertIn("鈴木玲子", prompt)
        self.assertNotIn(
            "钱二", self.store.load_registry(self.project["id"], "人名")["entries"]
        )

    def test_translation_writes_new_mapping_and_output(self):
        client = FakeModelClient(
            {
                "translated_text": "田村花子は東京で鈴木玲子と会った。",
                "new_replacements": [
                    {
                        "category": "人名",
                        "source": "赵一",
                        "target": "田村花子",
                        "notes": "首次出现",
                    },
                    {
                        "category": "人名",
                        "source": "钱二",
                        "target": "鈴木玲子",
                        "notes": "首次出现",
                    },
                    {
                        "category": "地名",
                        "source": "上海",
                        "target": "東京",
                        "notes": "目标文化城市",
                    },
                ],
                "warnings": [],
            }
        )
        engine = TranslationEngine(self.store, client)
        result = engine.translate_chapter(
            self.project["id"], self.chapter["id"], self.profile
        )

        manifest = self.store.load_project(self.project["id"])
        chapter_mappings = self.store.chapter_mappings(
            self.project["id"], self.chapter["id"]
        )
        registry = self.store.load_registry(self.project["id"], "人名")
        latest = self.store.read_latest_translation(
            self.project["id"], self.chapter["id"]
        )
        self.assertEqual(2, result["new_replacements"])
        self.assertEqual("田村花子", chapter_mappings["人名"]["赵一"]["target"])
        self.assertNotIn("赵一", registry["entries"])
        self.assertIn("田村花子は東京", latest)
        self.assertEqual("已翻译", manifest["chapters"][0]["status"])
        self.assertEqual(1, len(manifest["chapters"][0]["translations"]))

    def test_prompt_ignores_confirmed_global_entities_from_another_file(self):
        memory = translator_module.ProjectMemoryStore(self.store)
        memory._write({
            "schema_version": 1,
            "project_id": self.project["id"],
            "version": 0,
            "entities": [{
                "category": "人名", "source": "旧章角色", "target": "旧章名字",
                "notes": "其他独立小说", "details": {}, "status": "已确认",
                "source_chapters": ["other-chapter"],
            }],
            "chapter_memory": {},
            "preparation": {},
            "updated_at": "",
        })

        _, prompt, _ = PromptBuilder(self.store).build(
            self.project["id"], self.chapter["id"], mode="翻译",
            source_override="本章角色独自行走。",
        )

        self.assertNotIn("旧章角色", prompt)
        self.assertNotIn("旧章名字", prompt)

    def test_long_translation_passes_context_and_temporary_terms_forward(self):
        stored_source = (
            self.root
            / "projects"
            / self.project["id"]
            / "source"
            / self.chapter["stored_name"]
        )
        stored_source.write_bytes(
            "赵一登场。\n\n钱二继续。\n\n两人同行。".encode("utf-8")
        )
        responses = [
            json.dumps(
                {
                    "translated_text": "田村登場。\n\n",
                    "new_replacements": [
                        {
                            "category": "人名",
                            "source": "赵一",
                            "target": "田村",
                            "notes": "",
                        }
                    ],
                    "warnings": [],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "translated_text": "鈴木が続く。\n\n",
                    "new_replacements": [
                        {
                            "category": "人名",
                            "source": "钱二",
                            "target": "鈴木",
                            "notes": "",
                        }
                    ],
                    "warnings": ["第二段提示"],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "translated_text": "二人は同行した。",
                    "new_replacements": [],
                    "warnings": [],
                },
                ensure_ascii=False,
            ),
        ]
        client = SequencedModelClient(responses)

        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 8):
            result = TranslationEngine(self.store, client).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        self.assertEqual("田村登場。\n\n鈴木が続く。\n\n二人は同行した。", result["translated_text"])
        self.assertEqual(3, len(client.calls))
        self.assertNotIn("田村", client.calls[0]["user_prompt"])
        self.assertIn("田村", client.calls[1]["user_prompt"])
        self.assertIn("上一段只读上下文", client.calls[1]["user_prompt"])
        self.assertIn("田村登場", client.calls[1]["user_prompt"])
        self.assertIn("鈴木", client.calls[2]["user_prompt"])
        self.assertEqual(300, result["usage"]["input_tokens"])
        self.assertIn("第二段提示", result["warnings"])

    def test_japanese_rewrite_omits_context_at_natural_boundary_but_keeps_terms(self):
        self.stored_source_path.write_text(
            "セレンが来た。\n\nセレンは座った。", encoding="utf-8"
        )
        first = json.dumps(
            {
                "translated_text": "ミレイが来た。\n\n",
                "new_replacements": [
                    {
                        "category": "人名",
                        "source": "セレン",
                        "target": "ミレイ",
                        "notes": "",
                    }
                ],
                "memory_update": {"summary": "前段要約"},
                "warnings": [],
            },
            ensure_ascii=False,
        )
        second = json.dumps(
            {
                "translated_text": "ミレイは座った。",
                "new_replacements": [],
                "warnings": [],
            },
            ensure_ascii=False,
        )
        client = SequencedModelClient([first, second])

        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 11):
            TranslationEngine(self.store, client).translate_chapter(
                self.project["id"],
                self.chapter["id"],
                self.profile,
                mode="润色",
            )

        second_prompt = client.calls[1]["user_prompt"]
        self.assertIn("ミレイ", second_prompt)
        self.assertNotIn("上一段只读上下文", second_prompt)
        self.assertNotIn("セレンが来た。", second_prompt)
        self.assertNotIn("ミレイが来た。", second_prompt)
        self.assertNotIn("前段要約", second_prompt)

    def test_japanese_rewrite_hard_cut_sends_only_bounded_source_context(self):
        self.stored_source_path.write_text(
            "あ" * 250 + "い" * 250, encoding="utf-8"
        )
        response = json.dumps(
            {
                "translated_text": "前段の訳文",
                "new_replacements": [],
                "warnings": [],
            },
            ensure_ascii=False,
        )
        client = SequencedModelClient([response, response])

        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 250):
            TranslationEngine(self.store, client).translate_chapter(
                self.project["id"],
                self.chapter["id"],
                self.profile,
                mode="润色",
            )

        second_prompt = client.calls[1]["user_prompt"]
        self.assertIn("【硬切原文接缝】", second_prompt)
        self.assertIn("あ" * 200, second_prompt)
        self.assertNotIn("あ" * 201, second_prompt)
        self.assertNotIn("上一段译文末尾", second_prompt)
        self.assertNotIn("前段の訳文", second_prompt)

    def test_japanese_rewrite_does_not_rewrite_chapter_memory(self):
        memory_store = translator_module.ProjectMemoryStore(self.store)
        before = memory_store.update_chapter_memory(
            self.project["id"], self.chapter["id"], {"summary": "既存要約"}
        )
        client = FakeModelClient(
            {
                "translated_text": "田村は上海で銭と会った。",
                "new_replacements": [],
                "warnings": [],
            }
        )

        TranslationEngine(self.store, client).translate_chapter(
            self.project["id"],
            self.chapter["id"],
            self.profile,
            mode="润色",
        )

        after = memory_store.load(self.project["id"])
        self.assertEqual(before["version"], after["version"])

    def test_chinese_localization_uses_bounded_source_and_translation_context(self):
        self.stored_source_path.write_text(
            "很早的内容。\n\n赵一站起。\n\n钱二继续走。", encoding="utf-8"
        )
        first_translation = "旧" * 350 + "。田村は立った。\n\n"
        first = json.dumps(
            {
                "translated_text": first_translation,
                "new_replacements": [],
                "warnings": [],
            },
            ensure_ascii=False,
        )
        second = json.dumps(
            {
                "translated_text": "鈴木は歩き続けた。",
                "new_replacements": [],
                "warnings": [],
            },
            ensure_ascii=False,
        )
        client = SequencedModelClient([first, second])

        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 19):
            TranslationEngine(self.store, client).translate_chapter(
                self.project["id"],
                self.chapter["id"],
                self.profile,
                mode="翻译",
            )

        second_prompt = client.calls[1]["user_prompt"]
        self.assertIn("赵一站起。", second_prompt)
        self.assertNotIn("很早的内容。", second_prompt)
        self.assertIn("田村は立った。", second_prompt)
        self.assertNotIn("旧" * 20, second_prompt)
        self.assertIn("上一段只读上下文", second_prompt)

    def test_later_segment_failure_does_not_persist_earlier_terms(self):
        stored_source = (
            self.root
            / "projects"
            / self.project["id"]
            / "source"
            / self.chapter["stored_name"]
        )
        stored_source.write_text("赵一登场。\n\n钱二继续。", encoding="utf-8")
        first = json.dumps(
            {
                "translated_text": "田村登場。\n\n",
                "new_replacements": [
                    {
                        "category": "人名",
                        "source": "赵一",
                        "target": "田村",
                        "notes": "",
                    }
                ],
                "warnings": [],
            },
            ensure_ascii=False,
        )
        client = SequencedModelClient(
            [first, "not json", "still not json", "again not json"]
            + ["not json", "still not json", "again not json"]
        )

        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 8):
            with self.assertRaisesRegex(ValidationError, "第 2/2 段"):
                TranslationEngine(self.store, client).translate_chapter(
                    self.project["id"], self.chapter["id"], self.profile
                )

        self.assertIsNone(
            self.store.read_latest_translation(self.project["id"], self.chapter["id"])
        )
        self.assertNotIn(
            "赵一", self.store.load_registry(self.project["id"], "人名")["entries"]
        )

    def test_segment_progress_logs_do_not_include_source_text(self):
        secret_source = "不可写进日志甲。\n\n不可写进日志乙。"
        stored_source = (
            self.root
            / "projects"
            / self.project["id"]
            / "source"
            / self.chapter["stored_name"]
        )
        stored_source.write_text(secret_source, encoding="utf-8")
        payload = json.dumps(
            {"translated_text": "译文", "new_replacements": [], "warnings": []},
            ensure_ascii=False,
        )
        client = LoggingSequencedModelClient([payload, payload])

        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 10):
            TranslationEngine(self.store, client).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        combined = "\n".join(client.logs)
        self.assertIn("拆分为 2 段", combined)
        self.assertIn("分段 1/2", combined)
        self.assertIn("分段 2/2", combined)
        self.assertNotIn("不可写进日志", combined)

    def test_translation_repairs_invalid_json_with_embedded_quote(self):
        client = SequencedModelClient(
            [
                'Here is the result: {"translated_text":"bad "quote""}',
                '{"translated_text":"修复后的译文","new_replacements":[],"warnings":[]}',
            ]
        )

        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile
        )

        self.assertEqual("修复后的译文", result["translated_text"])
        self.assertEqual(2, len(client.calls))

    def test_truncated_segment_is_split_without_json_repair(self):
        stored_source = (
            self.root / "projects" / self.project["id"] / "source"
            / self.chapter["stored_name"]
        )
        stored_source.write_text("甲乙丙丁戊己庚辛", encoding="utf-8")
        payload = json.dumps(
            {"translated_text": "译文", "new_replacements": [], "warnings": []},
            ensure_ascii=False,
        )
        client = TruncatedThenSuccessClient([payload, payload, payload])

        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 100), patch(
            "novel_localizer.translator.MIN_SEGMENT_CHARS", 1
        ):
            result = TranslationEngine(self.store, client).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        self.assertEqual("译文\n\n译文", result["translated_text"])
        self.assertEqual(3, len(client.calls))
        self.assertNotIn("<model_response>", client.calls[1]["user_prompt"])

    def test_short_truncated_segment_retries_once_with_full_segment_budget(self):
        payload = json.dumps(
            {"translated_text": "译文", "new_replacements": [], "warnings": []},
            ensure_ascii=False,
        )
        client = TruncatedThenSuccessClient([payload, payload])

        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile
        )

        self.assertEqual("译文", result["translated_text"])
        self.assertEqual(2, len(client.calls))
        self.assertEqual(4_096, client.calls[0]["max_output_tokens"])
        self.assertEqual(12_000, client.calls[1]["max_output_tokens"])

    def test_same_language_rewrite_keeps_segment_after_exhausted_truncation_budget(self):
        same_language_project = self.store.create_project(
            "日语润色", "日语", "日语", ["人名"], []
        )
        same_language_chapter = self.store.import_chapters(
            same_language_project["id"], [self.chapter_path]
        ).imported[0]
        client = AlwaysTruncatedClient()

        result = TranslationEngine(self.store, client).translate_chapter(
            same_language_project["id"],
            same_language_chapter["id"],
            self.profile,
            mode="润色",
        )

        self.assertEqual("赵一在上海见到了钱二。", result["translated_text"])
        self.assertEqual(2, len(client.calls))
        self.assertEqual(4_096, client.calls[0]["max_output_tokens"])
        self.assertEqual(12_000, client.calls[1]["max_output_tokens"])
        self.assertTrue(any("保留原文" in item for item in result["warnings"]))

    def test_cross_language_translation_fails_after_exhausted_truncation_budget(self):
        with self.assertRaises(ValidationError):
            TranslationEngine(self.store, AlwaysTruncatedClient()).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile, mode="翻译"
            )

    def test_translation_repairs_invalid_json_without_resending_source(self):
        client = SequencedModelClient(
            [
                "not json",
                '{"translated_text":"修复后的译文","new_replacements":[],"warnings":[]}',
            ]
        )

        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile
        )

        self.assertEqual("修复后的译文", result["translated_text"])
        self.assertEqual(2, len(client.calls))
        self.assertIn("JSON 格式修复", client.calls[1]["system_prompt"])
        self.assertEqual("not json", client.calls[1]["user_prompt"])
        self.assertNotIn("赵一在上海", client.calls[1]["user_prompt"])
        self.assertEqual(200, result["usage"]["input_tokens"])
        self.assertTrue(any("自动修复 JSON" in warning for warning in result["warnings"]))

    def test_translation_does_not_save_when_regeneration_fails(self):
        client = SequencedModelClient(
            ["not json", "still not json", "again not json"]
            + ["not json", "still not json", "again not json"]
        )

        with self.assertRaises(ValidationError) as error:
            TranslationEngine(self.store, client).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        self.assertEqual(2, len(client.calls))
        self.assertIn("JSON 无法解析", str(error.exception))
        self.assertIsNone(
            self.store.read_latest_translation(self.project["id"], self.chapter["id"])
        )

    def test_later_segment_failure_keeps_a_checkpoint_for_resume(self):
        self.stored_source_path.write_text(
            "甲乙丙丁。\n\n戊己庚辛。", encoding="utf-8"
        )
        first = json.dumps(
            {"translated_text": "第一段译文", "new_replacements": [], "warnings": []},
            ensure_ascii=False,
        )
        client = SequencedModelClient(
            [first, "not json", "still not json", "again not json"]
            + ["not json", "still not json", "again not json"]
        )

        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 7), self.assertRaises(
            ValidationError
        ) as error:
            TranslationEngine(self.store, client).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        checkpoint = self.store.load_translation_checkpoint(
            self.project["id"], self.chapter["id"]
        )
        self.assertEqual(["第一段译文"], checkpoint["translated_segments"])
        self.assertIn("第 2/2 段", str(error.exception))
        self.assertEqual(3, len(client.calls))
        self.assertIsNone(
            self.store.read_latest_translation(self.project["id"], self.chapter["id"])
        )

    def test_retry_resumes_at_the_first_unfinished_segment(self):
        self.stored_source_path.write_text(
            "甲乙丙丁。\n\n戊己庚辛。", encoding="utf-8"
        )
        first = json.dumps(
            {"translated_text": "第一段译文", "new_replacements": [], "warnings": []},
            ensure_ascii=False,
        )
        failed = SequencedModelClient(
            [first, "not json", "still not json", "again not json"]
            + ["not json", "still not json", "again not json"]
        )
        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 7), self.assertRaises(
            ValidationError
        ):
            TranslationEngine(self.store, failed).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        resumed = SequencedModelClient([
            json.dumps(
                {"translated_text": "第二段译文", "new_replacements": [], "warnings": []},
                ensure_ascii=False,
            )
        ])
        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 7):
            result = TranslationEngine(self.store, resumed).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        self.assertEqual("第一段译文\n\n第二段译文", result["translated_text"])
        self.assertEqual(1, len(resumed.calls))
        self.assertIn("戊己庚辛", resumed.calls[0]["user_prompt"])
        self.assertIsNone(
            self.store.load_translation_checkpoint(self.project["id"], self.chapter["id"])
        )

    def test_changing_rename_policy_discards_partial_translation_checkpoint(self):
        self.stored_source_path.write_text(
            "甲乙丙丁。\n\n戊己庚辛。", encoding="utf-8"
        )
        first = json.dumps(
            {"translated_text": "旧策略第一段", "new_replacements": [], "warnings": []},
            ensure_ascii=False,
        )
        failed = SequencedModelClient(
            [first, "not json", "still not json", "again not json"]
            + ["not json", "still not json", "again not json"]
        )
        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 7), self.assertRaises(
            ValidationError
        ):
            TranslationEngine(self.store, failed).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        self.store.update_project_settings(
            self.project["id"], proper_noun_localization=False
        )
        restarted = SequencedModelClient([
            json.dumps(
                {"translated_text": "新策略第一段", "new_replacements": [], "warnings": []},
                ensure_ascii=False,
            ),
            json.dumps(
                {"translated_text": "新策略第二段", "new_replacements": [], "warnings": []},
                ensure_ascii=False,
            ),
        ])

        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 7):
            result = TranslationEngine(self.store, restarted).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        self.assertEqual("新策略第一段\n\n新策略第二段", result["translated_text"])
        self.assertEqual(2, len(restarted.calls))

    def test_mismatched_checkpoint_is_discarded_before_translation(self):
        self.stored_source_path.write_text(
            "甲乙丙丁。\n\n戊己庚辛。", encoding="utf-8"
        )
        self.store.save_translation_checkpoint(self.project["id"], self.chapter["id"], {
            "source_hash": "stale",
            "mode": "翻译",
            "segments": [{"text": "旧内容", "boundary_before": "start"}],
            "translated_segments": ["旧译文"],
        })
        client = FakeModelClient(
            {"translated_text": "新译文", "new_replacements": [], "warnings": []}
        )

        with patch("novel_localizer.translator.MAX_SEGMENT_CHARS", 100):
            TranslationEngine(self.store, client).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        self.assertIn("甲乙丙丁", client.calls[0]["user_prompt"])
        self.assertNotIn("旧内容", client.calls[0]["user_prompt"])
        self.assertIsNone(
            self.store.load_translation_checkpoint(self.project["id"], self.chapter["id"])
        )

    def test_global_mapping_does_not_constrain_an_independent_chapter(self):
        self.store.upsert_replacement(
            self.project["id"], "人名", "赵一", "田村花子"
        )
        client = FakeModelClient(
            {
                "translated_text": "山田花子は東京へ行った。",
                "new_replacements": [
                    {
                        "category": "人名",
                        "source": "赵一",
                        "target": "山田花子",
                        "notes": "",
                    }
                ],
                "warnings": [],
            }
        )
        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile
        )
        mapping = self.store.chapter_mappings(
            self.project["id"], self.chapter["id"]
        )["人名"]["赵一"]
        registry = self.store.load_registry(self.project["id"], "人名")

        self.assertEqual("山田花子", mapping["target"])
        self.assertEqual("田村花子", registry["entries"]["赵一"]["target"])
        self.assertFalse(result["warnings"])

    def test_mapping_not_in_source_is_rejected(self):
        client = FakeModelClient(
            {
                "translated_text": "翻訳結果",
                "new_replacements": [
                    {
                        "category": "人名",
                        "source": "不存在的人",
                        "target": "偽名",
                        "notes": "",
                    }
                ],
                "warnings": [],
            }
        )
        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile
        )
        registry = self.store.load_registry(self.project["id"], "人名")
        self.assertNotIn("不存在的人", registry["entries"])
        self.assertTrue(any("原文中未找到" in x for x in result["warnings"]))

    def test_parse_json_code_fence(self):
        parsed = parse_translation_json(
            '```json\n{"translated_text":"译文","new_replacements":[],"warnings":[]}\n```'
        )
        self.assertEqual("译文", parsed["translated_text"])

    def test_metadata_only_last_segment_can_be_empty_without_losing_body(self):
        self.stored_source_path.write_text(
            "正文。\n\n作者后记。", encoding="utf-8"
        )
        client = SequencedModelClient([
            json.dumps(
                {"translated_text": "处理后的正文。", "new_replacements": [], "warnings": []},
                ensure_ascii=False,
            ),
            json.dumps(
                {"translated_text": "", "new_replacements": [], "warnings": []},
                ensure_ascii=False,
            ),
        ])

        with patch(
            "novel_localizer.translator.split_source_segments",
            return_value=[
                translator_module.SourceSegment("正文。", "start"),
                translator_module.SourceSegment("作者后记。", "paragraph"),
            ],
        ):
            result = TranslationEngine(self.store, client).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile, mode="润色"
            )

        self.assertEqual("处理后的正文。", result["translated_text"])
        self.assertEqual(2, len(client.calls))

    def test_parser_allows_empty_translation_only_when_explicitly_requested(self):
        raw = '{"translated_text":"","new_replacements":[],"warnings":[]}'

        with self.assertRaises(ValidationError):
            parse_translation_json(raw)

        self.assertEqual(
            "", parse_translation_json(raw, allow_empty_translated_text=True)["translated_text"]
        )

    def test_invalid_translation_json_is_rejected(self):
        with self.assertRaises(ValidationError):
            parse_translation_json("不是 JSON")

    def test_invalid_translation_json_reports_bounded_nearby_content(self):
        malformed = '{"translated_text":"bad "quote""}'

        with self.assertRaisesRegex(ValidationError, "第 1 行第 26 列") as error:
            parse_translation_json(malformed)

        self.assertIn("附近内容", str(error.exception))
        self.assertIn('bad "quote"', str(error.exception))


if __name__ == "__main__":
    unittest.main()
