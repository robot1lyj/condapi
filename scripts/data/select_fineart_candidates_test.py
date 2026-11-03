"""Standard-library finite-language/decision tests; no Torch or model imports."""

# ruff: noqa: PT009, PT027
# Keep this suite executable with standard-library unittest, without pytest.

import copy
from fractions import Fraction
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "fineart_selector_under_test", Path(__file__).with_name("select_fineart_candidates.py")
)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):  # noqa: FBT002 - Match the tokenizer API.
        assert add_special_tokens is False
        return [ord(character) + 1 for character in text]

    def decode(self, tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False):  # noqa: FBT002 - Match the tokenizer API.
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token - 1) for token in tokens)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def make_task(offset=2.0):
    origin = round(offset * selector.FPS)
    videos = {
        camera: {
            "path": camera + ".mp4",
            "sha256": str(index + 1) * 64,
            "offset_s": offset,
            "parent_frame_zero_source_index": origin,
        }
        for index, camera in enumerate(selector.CAMERAS)
    }
    points = [10, 20, 29]
    frames = [
        {
            "frame_id": f"atom:{camera}:f{point}",
            "camera": camera,
            "parent_frame_index": point,
            "source_frame_index": origin + point,
            "video_sha256": videos[camera]["sha256"],
        }
        for point in points
        for camera in selector.CAMERAS
    ]
    task = {
        "task_id": "train:1:a0:select",
        "atom_id": "train:1:a0",
        "pool": "train",
        "source_episode_index": 1,
        "from_frame": 10,
        "to_frame": 30,
        "active_candidate_ids": ["C000"],
        "neighbor_candidate_ids": ["C001"],
        "candidates": [
            {"candidate_id": "C000", "caption": "Fold the sleeve.", "source_label_ids": ["raw:1"], "origin": "active"},
            {
                "candidate_id": "C001",
                "caption": "Flatten the shirt.",
                "source_label_ids": ["raw:2"],
                "origin": "neighbor",
            },
        ],
        "evidence": {"videos": videos, "parent_frame_indices": points, "frames": frames},
        "evidence_frame_ids": [frame["frame_id"] for frame in frames],
    }
    task["task_sha256"] = selector.task_sha(task)
    return task


class TrieTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.sequences, self.trie = selector.answer_language(self.tokenizer, ["C000", "C001"], 0)

    def test_finite_language_contains_only_offered_ids_and_unknown(self):
        complete = set()

        def visit(prefix):
            if prefix in self.trie.sequences:
                complete.add(selector.decode_answer(prefix, self.sequences))
                return
            self.assertLess(len(prefix), self.trie.max_length)
            for token in self.trie.allowed(prefix):
                visit((*prefix, token))

        visit(())
        self.assertEqual(complete, {"C000", "C001", "UNKNOWN"})

    def test_shared_token_prefix_keeps_only_valid_branches(self):
        prefix = self.tokenizer.encode("C00")
        self.assertEqual(self.trie.allowed(prefix), self.tokenizer.encode("01"))
        self.assertEqual(self.trie.allowed(self.tokenizer.encode("C000")), [0])

    def test_shorter_answer_can_end_or_continue_to_longer_offered_answer(self):
        _, trie = selector.answer_language(self.tokenizer, ["C000", "C0000"], 0)
        self.assertEqual(trie.allowed(self.tokenizer.encode("C000")), [0, ord("0") + 1])

    def test_illegal_prefix_and_post_eos_continuation_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "escaped"):
            self.trie.allowed(self.tokenizer.encode("NEW_ACTION"))
        with self.assertRaisesRegex(ValueError, "No continuation"):
            self.trie.allowed(self.sequences["C000"])

    def test_callback_uses_only_generated_suffix_and_enforces_batch_one(self):
        callback = selector.prefix_allowed_tokens_fn(self.trie, [123, 456])
        self.assertEqual(callback(0, FakeTensor([123, 456])), self.trie.allowed([]))
        self.assertEqual(
            callback(0, FakeTensor([123, 456, *self.tokenizer.encode("C00")])),
            self.trie.allowed(self.tokenizer.encode("C00")),
        )
        with self.assertRaisesRegex(ValueError, "batch size one"):
            callback(1, FakeTensor([123, 456]))
        with self.assertRaisesRegex(ValueError, "prompt prefix changed"):
            callback(0, FakeTensor([123, 999]))

    def test_exact_answer_requires_eos_and_cannot_add_explanations(self):
        for answer in ("C000", "C001", "UNKNOWN"):
            self.assertEqual(selector.decode_answer(self.sequences[answer], self.sequences), answer)
        for tokens in (
            self.sequences["C000"][:-1],
            self.sequences["C000"] + (12,),
            (*tuple(self.tokenizer.encode("C002")), 0),
        ):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, "incomplete or escaped"):
                selector.decode_answer(tokens, self.sequences)

    def test_duplicate_reserved_and_colliding_answer_tokens_are_rejected(self):
        for candidates in (["C000", "C000"], ["UNKNOWN"]):
            with self.subTest(candidates=candidates), self.assertRaisesRegex(ValueError, "unique"):
                selector.answer_language(self.tokenizer, candidates, 0)

        class CollidingTokenizer(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):  # noqa: FBT002 - Match the tokenizer API.
                self.latest = text
                return [9]

            def decode(self, tokens, **kwargs):
                return self.latest

        with self.assertRaisesRegex(ValueError, "identical token sequences"):
            selector.answer_language(CollidingTokenizer(), ["C000"], 0)

    def test_empty_candidate_set_still_offers_unknown(self):
        sequences, trie = selector.answer_language(self.tokenizer, [], 0)
        self.assertEqual(set(sequences), {"UNKNOWN"})
        self.assertEqual(selector.decode_answer(sequences["UNKNOWN"], sequences), "UNKNOWN")
        self.assertGreater(trie.max_length, 0)

    def test_tokenizer_round_trip_and_eos_inside_answer_are_checked(self):
        class BadRoundTrip(FakeTokenizer):
            def decode(self, tokens, **kwargs):
                return "different"

        class EmbeddedEos(FakeTokenizer):
            def encode(self, text, **kwargs):
                return [0]

        with self.assertRaisesRegex(ValueError, "round trip"):
            selector.answer_language(BadRoundTrip(), ["C000"], 0)
        with self.assertRaisesRegex(ValueError, "contains EOS"):
            selector.answer_language(EmbeddedEos(), ["C000"], 0)


class TaskAndReceiptTests(unittest.TestCase):
    def test_module_and_help_do_not_import_torch_or_transformers(self):
        with tempfile.TemporaryDirectory(prefix="fineart-lazy-import-test-") as temporary:
            root = Path(temporary)
            for name in ("torch", "transformers"):
                (root / (name + ".py")).write_text("raise AssertionError('unexpected model runtime import')\n")
            environment = dict(os.environ, PYTHONPATH=str(root))
            code = (
                "import importlib.util,sys; "
                f"spec=importlib.util.spec_from_file_location('selector_lazy_check',{str(Path(SPEC.origin))!r}); "
                "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
            )
            imported = subprocess.run(
                [sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=10, check=False
            )
            self.assertEqual(imported.returncode, 0, imported.stderr)
            help_result = subprocess.run(
                [sys.executable, str(Path(SPEC.origin)), "--help"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("--tasks", help_result.stdout)

    def test_task_hash_binds_caption_boundaries_and_exact_evidence(self):
        task = make_task()
        selector.validate_task(task)
        for edit in (
            lambda value: value.update(from_frame=11),
            lambda value: value["candidates"][0].update(caption="New action"),
            lambda value: value["evidence_frame_ids"].reverse(),
        ):
            changed = copy.deepcopy(task)
            edit(changed)
            with self.subTest(edit=edit), self.assertRaisesRegex(ValueError, "Task SHA-256"):
                selector.validate_task(changed)

    def test_missing_view_and_wrong_source_offset_fail_even_with_new_task_hash(self):
        for edit in (
            lambda value: value["evidence"]["frames"].pop(),
            lambda value: value["evidence"]["videos"]["top"].update(parent_frame_zero_source_index=61),
        ):
            task = make_task()
            edit(task)
            task["task_sha256"] = selector.task_sha(task)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                selector.validate_task(task)

    def test_parent_preroll_and_positive_negative_float_residues_are_bound(self):
        for offset in (2.0, 2.0 + 5.68e-14, 2.0 - 5.68e-14):
            task = make_task(offset)
            selector.validate_task(task)
            extracted = {
                (frame["video_sha256"], frame["source_frame_index"]): {
                    "path": "fake.png",
                    "sha256": "a" * 64,
                    "actual_decoded_source_index": frame["source_frame_index"],
                    "source_pts_seconds": str(Fraction(frame["source_frame_index"], 30)),
                }
                for frame in task["evidence"]["frames"]
            }
            images = selector.task_evidence(task, extracted)
            self.assertEqual([image["frame_id"] for image in images], task["evidence_frame_ids"])
            for image in images:
                self.assertEqual(image["source_frame_index"], image["parent_frame_index"] + 60)
                self.assertLess(
                    abs(Fraction(image["parent_relative_pts_seconds"]) - Fraction(image["parent_frame_index"], 30)),
                    Fraction(1, 100000),
                )
            key = next(iter(extracted))
            extracted[key]["source_pts_seconds"] = str(Fraction(key[1] + 1, 30))
            with self.assertRaisesRegex(ValueError, "parent offset/frame clock"):
                selector.task_evidence(task, extracted)

    def test_decision_contract_and_resume_reject_changed_evidence(self):
        task = make_task()
        identity = selector.selector_identity()
        self.assertEqual(set(identity), {"model_id", "model_revision", "implementation_version", "prompt_sha256"})
        with tempfile.TemporaryDirectory(prefix="fineart-receipt-test-") as temporary:
            root = Path(temporary)
            image = root / "evidence.png"
            image.write_bytes(b"synthetic protocol evidence, not a decoded image")
            receipt = {
                "binding_sha256": "b" * 64,
                "selector": identity,
                "task_id": task["task_id"],
                "atom_id": task["atom_id"],
                "task_sha256": task["task_sha256"],
                "selected_id": "C000",
                "input_video_sha256": {
                    camera: task["evidence"]["videos"][camera]["sha256"] for camera in selector.CAMERAS
                },
                "evidence": [
                    {"frame_id": frame_id, "path": str(image), "sha256": selector.digest(image)}
                    for frame_id in task["evidence_frame_ids"]
                ],
            }
            path = root / "receipt.json"
            selector.publish(path, selector.json_bytes(receipt))
            decision = selector.recover(task, path, identity, "b" * 64)
            self.assertEqual(
                set(decision),
                {
                    "task_id",
                    "atom_id",
                    "task_sha256",
                    "selected_id",
                    "evidence_frame_ids",
                    "input_video_sha256",
                    "inference_receipt",
                },
            )
            self.assertEqual(decision["selected_id"], "C000")
            image.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "evidence image changed"):
                selector.recover(task, path, identity, "b" * 64)

    def test_saved_envelope_is_accepted_by_actual_timeline_decision_reader(self):
        spec = importlib.util.spec_from_file_location(
            "fineart_timeline_reader_for_selector_test", Path(__file__).with_name("auto_fineart_timeline.py")
        )
        timeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(timeline)
        task = make_task()
        identity = selector.selector_identity()
        plan = {
            "schema_version": selector.PLAN_SCHEMA,
            "tasks": [task],
            "policy_contract": {
                "models": {"selector": {"model_id": selector.MODEL_ID, "model_revision": selector.MODEL_REVISION}}
            },
        }
        with tempfile.TemporaryDirectory(prefix="fineart-envelope-api-test-") as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            plan_path.write_bytes(selector.json_bytes(plan))
            image = root / "synthetic-protocol-evidence.png"
            image.write_bytes(b"not a real decoded image; protocol fixture only")
            receipt = {
                "selector": identity,
                "task_id": task["task_id"],
                "atom_id": task["atom_id"],
                "task_sha256": task["task_sha256"],
                "selected_id": "UNKNOWN",
                "input_video_sha256": {
                    camera: task["evidence"]["videos"][camera]["sha256"] for camera in selector.CAMERAS
                },
                "evidence": [
                    {"frame_id": frame_id, "path": str(image), "sha256": selector.digest(image)}
                    for frame_id in task["evidence_frame_ids"]
                ],
            }
            path = selector.receipt_path(root, task)
            selector.publish(path, selector.json_bytes(receipt))
            decision = selector.decision_from_receipt(task, receipt, path)
            selector.save_decisions(root, plan, selector.digest(plan_path), identity, {task["task_id"]: decision})
            choices, _ = timeline.read_decisions(plan_path, plan, root / "decisions.json")
            self.assertEqual(choices[task["task_id"]]["selected_id"], "UNKNOWN")
            self.assertEqual(choices[task["task_id"]]["evidence_frame_ids"], task["evidence_frame_ids"])

    def test_publication_is_readable_and_exclusive_under_restrictive_umask(self):
        with tempfile.TemporaryDirectory(prefix="fineart-publication-test-") as temporary:
            path = Path(temporary) / "receipt.json"
            previous = os.umask(0o077)
            try:
                selector.publish(path, b"first")
            finally:
                os.umask(previous)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
            with self.assertRaises(FileExistsError):
                selector.publish(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main()
