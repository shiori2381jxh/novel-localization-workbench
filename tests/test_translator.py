from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            mode="中文小说 → 日语本地化",
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

    def test_compatible_empty_content_reports_safe_response_diagnostics(self):
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

        with self.assertRaises(ModelAPIError) as raised:
            client.generate(
                profile,
                "不能写入日志的系统提示词",
                "不能写入日志的小说正文",
                max_output_tokens=2000,
            )

        message = str(raised.exception)
        self.assertIn("finish_reason=length", message)
        self.assertIn("response_id=chatcmpl_empty_1", message)
        self.assertIn("completion_tokens=2000", message)
        self.assertIn("message字段=content, reasoning_content, role", message)
        combined_logs = "\n".join(logs)
        self.assertIn("发送请求", combined_logs)
        self.assertIn("收到响应：HTTP 200", combined_logs)
        self.assertNotIn("do-not-log", combined_logs)
        self.assertNotIn("不能写入日志的系统提示词", combined_logs)
        self.assertNotIn("不能写入日志的小说正文", combined_logs)
        self.assertNotIn("不能写入日志的内部内容", combined_logs)


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
        self.store.upsert_replacement(
            self.project["id"], "人名", "赵一", "田村 花子"
        )
        _, prompt, source = PromptBuilder(self.store).build(
            self.project["id"],
            self.chapter["id"],
            mode="中文小说 → 日语本地化",
        )
        self.assertIn("田村 花子", prompt)
        self.assertIn("医院名称也要本地化", prompt)
        self.assertIn("赵一在上海", source)

    def test_modes_are_explicit_and_unknown_mode_is_rejected(self):
        builder = PromptBuilder(self.store)
        _, localization_prompt, _ = builder.build(
            self.project["id"], self.chapter["id"],
            mode="中文小说 → 日语本地化",
        )
        _, rewrite_prompt, _ = builder.build(
            self.project["id"], self.chapter["id"],
            mode="日语原文 → 改名润色",
        )

        self.assertIn("把中文正文翻译为自然日语", localization_prompt)
        self.assertIn("输入和输出均为日语", rewrite_prompt)
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
            mode="中文小说 → 日语本地化",
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
            mode="中文小说 → 日语本地化",
        )

        prompt = client.calls[0]["user_prompt"]
        self.assertIn("医院名称也要本地化", prompt)
        self.assertIn("中文小说 → 日语本地化", prompt)
        self.assertEqual("已确认", result["entities"][0]["status"])
        self.assertEqual("田村花子", result["entities"][0]["target"])

    def test_translation_starts_without_separate_entity_preparation(self):
        client = FakeModelClient(
            {"translated_text": "田村花子は東京にいる。", "new_replacements": [], "warnings": []}
        )
        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile,
            mode="中文小说 → 日语本地化", allow_unprepared=False,
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
            mode="日语原文 → 改名润色",
        )

        self.assertEqual(1, result["new_replacements"])
        registry = self.store.load_registry(self.project["id"], "家族名")
        self.assertEqual("ベルシュタイン", registry["entries"]["アルハイム"]["target"])

    def test_residual_forced_name_requires_review_without_automatic_retry(self):
        self.store.upsert_replacement(
            self.project["id"], "人名", "赵一", "田村花子"
        )
        client = FakeModelClient(
            {"translated_text": "赵一は東京にいる。", "new_replacements": [], "warnings": []}
        )
        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile,
            mode="中文小说 → 日语本地化", allow_unprepared=True,
        )
        self.assertEqual("完成", result["review_status"])
        self.assertEqual([], result["residual_sources"])
        self.assertFalse(result["can_retry"])
        self.assertEqual(1, len(client.calls))
        self.assertEqual("中文小说 → 日语本地化", result["record"]["mode"])
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
            warnings=["旧版误报"], mode="日语原文 → 改名润色",
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
            mode="日语原文 → 改名润色", allow_unprepared=True,
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
            mode="中文小说 → 日语本地化",
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

    def test_segment_prompt_includes_read_only_context_and_temporary_terms(self):
        _, prompt, source = PromptBuilder(self.store).build(
            self.project["id"],
            self.chapter["id"],
            mode="中文小说 → 日语本地化",
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
        registry = self.store.load_registry(self.project["id"], "人名")
        latest = self.store.read_latest_translation(
            self.project["id"], self.chapter["id"]
        )
        self.assertEqual(3, result["new_replacements"])
        self.assertEqual("田村花子", registry["entries"]["赵一"]["target"])
        self.assertIn("田村花子は東京", latest)
        self.assertEqual("已翻译", manifest["chapters"][0]["status"])
        self.assertEqual(1, len(manifest["chapters"][0]["translations"]))

    def test_long_translation_passes_context_and_temporary_terms_forward(self):
        stored_source = (
            self.root
            / "projects"
            / self.project["id"]
            / "source"
            / self.chapter["stored_name"]
        )
        stored_source.write_text("赵一登场。\n\n钱二继续。\n\n两人同行。", encoding="utf-8")
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

    def test_translation_repairs_invalid_json_once(self):
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
        self.assertTrue(any("自动修复" in item for item in result["warnings"]))
        self.assertIn("bad \"quote\"", client.calls[1]["user_prompt"])

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

    def test_translation_regenerates_once_when_json_repair_fails(self):
        client = SequencedModelClient(
            [
                "not json",
                "still not json",
                '{"translated_text":"重生成译文","new_replacements":[],"warnings":[]}',
            ]
        )

        result = TranslationEngine(self.store, client).translate_chapter(
            self.project["id"], self.chapter["id"], self.profile
        )

        self.assertEqual("重生成译文", result["translated_text"])
        self.assertEqual(3, len(client.calls))
        self.assertIn("只返回一个有效 JSON 对象", client.calls[2]["user_prompt"])
        self.assertNotIn("<model_response>", client.calls[2]["user_prompt"])

    def test_translation_does_not_save_when_regeneration_fails(self):
        client = SequencedModelClient(["not json", "still not json", "again not json"])

        with self.assertRaises(ValidationError) as error:
            TranslationEngine(self.store, client).translate_chapter(
                self.project["id"], self.chapter["id"], self.profile
            )

        self.assertEqual(3, len(client.calls))
        self.assertIn("首次错误", str(error.exception))
        self.assertIn("自动修复后仍失败", str(error.exception))
        self.assertIsNone(
            self.store.read_latest_translation(self.project["id"], self.chapter["id"])
        )

    def test_conflicting_mapping_is_not_overwritten(self):
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
        registry = self.store.load_registry(self.project["id"], "人名")

        self.assertEqual("田村花子", registry["entries"]["赵一"]["target"])
        self.assertTrue(any("不能自动产生第二个名字" in x for x in result["warnings"]))

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
