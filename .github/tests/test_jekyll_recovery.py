"""Offline regression checks for the Jekyll cache and edition lookup."""

import http.server
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
RESTORE = ROOT / "docs/_scripts/restore_jekyll_cache.sh"
RUBY_HELPER = ROOT / "docs/_scripts/remote_editions.rb"


class RestoreCacheTests(unittest.TestCase):
    def run_restore(self, archive=False, extraction_succeeds=True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "github-output"
            output.touch()
            if archive:
                (root / "jekyll-content.zip").touch()
            bin_dir = root / "bin"
            bin_dir.mkdir()
            seven_zip = bin_dir / "7z"
            seven_zip.write_text(
                "#!/usr/bin/env bash\n"
                + (
                    "mkdir -p _site\n"
                    "touch _site/.jekyll-metadata _site/backup-models.json "
                    "_site/backup-benchmarking.json _site/backup-references.json\n"
                    if extraction_succeeds else "exit 2\n"
                )
            )
            seven_zip.chmod(0o755)
            result = subprocess.run(
                ["bash", str(RESTORE)], cwd=root, capture_output=True, text=True,
                env={**os.environ, "GITHUB_OUTPUT": str(output),
                     "PATH": f"{bin_dir}:{os.environ['PATH']}"},
            )
            restored = output.read_text()
            metadata = (root / ".jekyll-metadata").exists()
            zip_exists = (root / "jekyll-content.zip").exists()
            return result, restored, metadata, zip_exists

    def test_no_archive_requests_full_build_without_error(self):
        result, restored, metadata, _ = self.run_restore()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("restored=false", restored)
        self.assertFalse(metadata)

    def test_archive_restores_incremental_metadata(self):
        result, restored, metadata, zip_exists = self.run_restore(archive=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("restored=true", restored)
        self.assertTrue(metadata)
        self.assertFalse(zip_exists)

    def test_corrupt_archive_fails_instead_of_masking_error(self):
        result, restored, _, _ = self.run_restore(archive=True, extraction_succeeds=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("restored=true", restored)


class RemoteEditionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/slow":
                    time.sleep(0.5)
                self.send_response(503 if self.path == "/error" else 200)
                self.end_headers()
                try:
                    payload = (b"null" if self.path == "/invalid" else
                               b'{"meta":{"aggregations":{"editions":["Spark NLP 6.2"]}}}')
                    self.wfile.write(payload)
                except BrokenPipeError:
                    pass  # The timeout case closes the client before the response arrives.

            def log_message(self, format, *args):
                pass

        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def fetch(self, route, timeout: float = 1):
        url = f"http://127.0.0.1:{self.server.server_port}/{route}"
        code = "puts RemoteEditions.fetch(ARGV.fetch(0), timeout_seconds: ARGV.fetch(1).to_f).to_a.join(',')"
        return subprocess.run(
            ["ruby", "-I", str(RUBY_HELPER.parent), "-rremote_editions",
             "-e", code, url, str(timeout)], capture_output=True, text=True,
        )

    def test_successful_response_returns_editions(self):
        result = self.fetch("ok")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "Spark NLP 6.2")

    def test_http_failure_is_not_mistaken_for_new_edition(self):
        result = self.fetch("error")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HTTP 503", result.stderr)

    def test_invalid_response_fails_instead_of_triggering_full_build(self):
        result = self.fetch("invalid")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid response", result.stderr)

    def test_slow_response_fails_within_request_deadline(self):
        result = self.fetch("slow", timeout=0.1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("timed out", result.stderr)


class WorkflowTests(unittest.TestCase):
    def test_missing_cache_skips_incremental_build_and_runs_full_build(self):
        workflow = (ROOT / ".github/workflows/create_search_index.yml").read_text()
        self.assertIn("JEKYLL_POST_BATCH", workflow)
        self.assertIn("workflow_run:", workflow)
        self.assertNotIn("rm -f .jekyll-metadata", workflow)
        self.assertIn("jekyll_wave_complete", workflow)

    def test_search_plugin_uses_bounded_remote_edition_fetch(self):
        plugin = (ROOT / "docs/_plugins/search_index.rb").read_text()
        self.assertIn("RemoteEditions.fetch(SEARCH_URL)", plugin)
        self.assertNotIn("Net::HTTP.get_response(uri)", plugin)

    def test_search_plugin_logs_progress_during_post_render(self):
        plugin = (ROOT / "docs/_plugins/search_index.rb").read_text()
        self.assertIn("Search index progress:", plugin)
        self.assertIn("$stdout.flush", plugin)
        self.assertIn("Search index post_render:", plugin)


if __name__ == "__main__":
    unittest.main()
