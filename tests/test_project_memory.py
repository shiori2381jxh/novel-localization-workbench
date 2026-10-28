from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from novel_localizer.domain import ValidationError
from novel_localizer.project_memory import ProjectMemoryStore
from novel_localizer.storage import ProjectStore


class ProjectMemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temp.name))
        self.project = self.store.create_project(
            "日语改写", "日语", "日语", ["人名", "地名", "第一人称"]
        )
        self.memory = ProjectMemoryStore(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_concurrent_chapter_memory_updates_do_not_overwrite_each_other(self):
        original_load = self.memory.load
        barrier = threading.Barrier(2)

        def synchronized_load(project_id):
            memory = original_load(project_id)
            try:
                barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
            return memory

        self.memory.load = synchronized_load
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        self.memory.update_chapter_memory,
                        self.project["id"], chapter_id, value,
                    )
                    for chapter_id, value in (
                        ("chapter-1", {"summary": "一"}),
                        ("chapter-2", {"summary": "二"}),
                    )
                ]
                for future in futures:
                    future.result()
        finally:
            self.memory.load = original_load

        self.assertEqual(
            {"chapter-1", "chapter-2"},
            set(self.memory.load(self.project["id"])["chapter_memory"]),
        )

    def test_candidates_require_confirmation_and_increment_version(self):
        saved = self.memory.save_candidates(
            self.project["id"], ["chapter-1"],
            [{"category": "人名", "source": "セレン", "target": "ミレイ", "notes": "主角"}],
        )
        self.assertEqual("待确认", saved["entities"][0]["status"])
        self.assertEqual(1, saved["version"])

        confirmed = self.memory.confirm_candidates(
            self.project["id"],
            [{"category": "人名", "source": "セレン", "target": "ミレイ", "notes": "女主角"}],
        )
        self.assertEqual("已确认", confirmed["entities"][0]["status"])
        self.assertEqual(2, confirmed["version"])

    def test_forced_identity_candidate_cannot_be_confirmed(self):
        self.memory.save_candidates(
            self.project["id"], ["chapter-1"],
            [{"category": "人名", "source": "セレン", "target": "セレン"}],
        )
        with self.assertRaisesRegex(ValidationError, "必须使用不同的新名称"):
            self.memory.confirm_candidates(
                self.project["id"],
                [{"category": "人名", "source": "セレン", "target": "セレン"}],
            )

    def test_non_forced_identity_candidate_can_be_confirmed(self):
        self.memory.save_candidates(
            self.project["id"], ["chapter-1"],
            [{"category": "第一人称", "source": "私", "target": "私"}],
        )
        result = self.memory.confirm_candidates(
            self.project["id"],
            [{"category": "第一人称", "source": "私", "target": "私"}],
        )
        self.assertEqual("已确认", result["entities"][0]["status"])

    def test_confirmation_validates_all_edits_before_writing_any_mapping(self):
        self.memory.save_candidates(
            self.project["id"], ["chapter-1"],
            [
                {"category": "人名", "source": "セレン", "target": "ミレイ"},
                {"category": "地名", "source": "王都", "target": "王都"},
            ],
        )
        with self.assertRaises(ValidationError):
            self.memory.confirm_candidates(
                self.project["id"],
                [
                    {"category": "人名", "source": "セレン", "target": "ミレイ"},
                    {"category": "地名", "source": "王都", "target": "王都"},
                ],
            )
        registry = self.store.load_registry(self.project["id"], "人名")
        self.assertNotIn("セレン", registry["entries"])

    def test_user_confirmation_can_change_an_existing_confirmed_name(self):
        self.store.upsert_replacement(
            self.project["id"], "人名", "セレン", "ミレイ"
        )
        self.memory.save_candidates(
            self.project["id"], ["chapter-1"],
            [{"category": "人名", "source": "セレン", "target": "ミレイ"}],
        )
        result = self.memory.confirm_candidates(
            self.project["id"],
            [{"category": "人名", "source": "セレン", "target": "エリナ"}],
        )
        self.assertEqual("エリナ", result["entities"][0]["target"])
        registry = self.store.load_registry(self.project["id"], "人名")
        self.assertEqual("エリナ", registry["entries"]["セレン"]["target"])

    def test_confirmation_creates_missing_entity_category_for_existing_project(self):
        self.memory.save_candidates(
            self.project["id"], ["chapter-1"],
            [{"category": "家族名", "source": "アルハイム", "target": "ベルシュタイン"}],
        )

        result = self.memory.confirm_candidates(
            self.project["id"],
            [{"category": "家族名", "source": "アルハイム", "target": "ベルシュタイン"}],
        )

        self.assertEqual("已确认", result["entities"][0]["status"])
        registry = self.store.load_registry(self.project["id"], "家族名")
        self.assertEqual(
            "ベルシュタイン", registry["entries"]["アルハイム"]["target"]
        )
