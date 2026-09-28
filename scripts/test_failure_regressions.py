"""Regressions for PDF parsing, interrupted audio, and CAS login bounces."""

import shutil
import subprocess
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.ai.transcriber import (
    EmptyTranscriptError,
    IncompleteAudioError,
    Transcriber,
)
from src.ai.transcript_proofreader import TranscriptProofreader
from src.api.webvpn import WebVPNLoginBounceError, WebVPNSession
from src.api.emailer import Emailer
from src.pipeline.blackboard_lecture_runner import BlackboardLectureRunner
from src.pdf.latex_renderer import _PANDOC_INPUT_FORMAT
from src.runtime.scheduler import Scheduler


class NoSpeechVAD:
    def accept_waveform(self, samples):
        pass

    def empty(self):
        return True

    def flush(self):
        pass


class FailureRegressions(unittest.TestCase):
    @unittest.skipUnless(shutil.which("pandoc"), "Pandoc is not installed")
    def test_bad_yaml_in_generated_notes_is_markdown(self):
        notes = "---\ntitle: |bad\n---\n正文 $x^2$\n"
        result = subprocess.run(
            ["pandoc", f"--from={_PANDOC_INPUT_FORMAT}", "--to=latex"],
            input=notes, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("x^2", result.stdout)
        self.assertIn("title:", result.stdout)

    def test_incomplete_and_empty_audio_are_retryable_errors(self):
        asr = Transcriber.__new__(Transcriber)
        asr._vad = NoSpeechVAD()
        chunk = np.ones(16000, dtype=np.float32).tobytes()

        def consume(reported_duration):
            chunks = iter([chunk, b""])
            with patch.object(asr, "_init"), patch.object(asr, "_reset_vad"):
                return asr._consume_pcm_stream(
                    read_fn=lambda n: next(chunks),
                    is_eof_fn=lambda: True,
                    stderr_provider=lambda: (
                        f"Duration: 00:00:{reported_duration:05.2f}".encode()
                    ),
                    return_code_fn=lambda: 0,
                    timeout=10,
                )

        with self.assertRaises(IncompleteAudioError):
            consume(3.0)
        with self.assertRaisesRegex(EmptyTranscriptError, "sampled peak=1.00000"):
            consume(1.0)

    def test_cas_login_bounce_renews_session(self):
        vpn = WebVPNSession.__new__(WebVPNSession)

        class LoginPage:
            url = "https://webvpn.fudan.edu.cn:443/login"
            status_code = 200
            headers = {}
            text = "Please log in"

        class Session:
            calls = 0

            def get(self, *args, **kwargs):
                self.calls += 1
                return LoginPage()

        vpn.session = Session()
        with self.assertRaises(WebVPNLoginBounceError):
            vpn.authenticate_icourse("student", "password")
        self.assertEqual(vpn.session.calls, 1)

    def test_short_gemini_quota_delay_does_not_disable_model(self):
        proofreader = TranscriptProofreader.__new__(TranscriptProofreader)
        proofreader._next_request_at = {}
        proofreader._disabled_models = {}
        proofreader._model_failure_counts = {}
        proofreader._circuit_failures = 2
        proofreader._record_model_failure(
            "gemini/gemini-3.6-flash",
            RuntimeError("429 quota exceeded. Please retry in 14.05s."),
        )
        self.assertNotIn("gemini/gemini-3.6-flash", proofreader._disabled_models)
        self.assertIn("gemini/gemini-3.6-flash", proofreader._next_request_at)

    def test_confirmed_silent_lecture_produces_visual_notes_without_asr(self):
        runner = BlackboardLectureRunner.__new__(BlackboardLectureRunner)
        runner._active_course_title = ""
        runner._active_course_id = ""
        runner._client = object()
        runner._reporter = MagicMock()
        runner._db = MagicMock()
        runner._db.get_lecture.return_value = None
        runner._db.get_done_ppt_pages.return_value = [
            {"page_num": 1, "created_sec": 90, "text": "函数空间与投影"}
        ]
        runner._ppt = MagicMock()
        runner._scheduler = MagicMock()
        with patch.object(runner, "_ensure_raw_blackboard", return_value=(
            "板书中的证明", "vision-test"
        )), patch.object(runner, "_edit_blackboard_or_fallback", return_value=(
            "### 黑板板书整理稿\n\n证明内容", "editor-test"
        )):
            notes = runner.run("course", "泛函分析", {
                "sub_id": "669978", "sub_title": "第3-5节", "date": "2026-09-28"
            })
        self.assertIn("没有人声", notes)
        self.assertIn("证明内容", notes)
        self.assertIn("函数空间与投影", notes)
        self.assertNotIn("AI 课程总结（带视频定位）", notes)
        self.assertTrue(runner._db.update_summary.call_args.args[2].startswith(
            "blackboard-llm-editor-v9/visual-only/no-speech"
        ))
        runner._db.mark_processed.assert_called_once_with("669978")
        runner._scheduler.audio_downloader.schedule.assert_not_called()
        mailer = Emailer.__new__(Emailer)
        self.assertEqual(mailer._complete_transcript({"visual_only": True}), "")

    def test_silent_prefetch_does_not_download_audio(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.image_cache = MagicMock()
        scheduler.audio_downloader = MagicMock()
        scheduler.prefetch_lecture(object(), "course", "669978")
        scheduler.image_cache.schedule.assert_called_once()
        scheduler.audio_downloader.schedule.assert_not_called()


if __name__ == "__main__":
    unittest.main()
