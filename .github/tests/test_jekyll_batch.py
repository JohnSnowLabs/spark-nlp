"""Offline checks for batched Jekyll recovery waves."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LIMITER = ROOT / "docs/_scripts/batch_limiter.rb"
PLUGIN = ROOT / "docs/_plugins/search_index.rb"
WORKFLOW = ROOT / ".github/workflows/create_search_index.yml"


class BatchLimiterTests(unittest.TestCase):
    def run_ruby(self, code, env):
        return subprocess.run(
            ["ruby", "-I", str(LIMITER.parent), "-rbatch_limiter", "-e", code],
            capture_output=True, text=True, env=env,
        )

    def test_defers_posts_beyond_the_wave_cap_without_recording_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "github-output"
            env = {
                **os.environ,
                "JEKYLL_POST_BATCH": "2",
                "JEKYLL_WAVE_NO_EXIT": "1",
                "GITHUB_OUTPUT": str(output),
            }
            code = """
posts = ['a.md', 'b.md', 'c.md']
recorded = []
result = posts.select do |path|
  BatchLimiter.allow?(path) { recorded << path }
end
puts result.join(',')
puts recorded.join(',')
BatchLimiter.write_status
puts BatchLimiter.deferred?
"""
            result = self.run_ruby(code, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered, recorded, deferred = result.stdout.splitlines()
            self.assertEqual(rendered, "a.md,b.md")
            self.assertEqual(recorded, "a.md,b.md")
            self.assertEqual(deferred, "true")
            self.assertIn("jekyll_wave_complete=false", output.read_text())

    def test_wave_cap_exits_the_build_instead_of_continuing(self):
        source = LIMITER.read_text()
        self.assertIn("exit 0", source)
        self.assertIn("finish_wave", source)
        workflow = WORKFLOW.read_text()
        self.assertIn("actions/upload-artifact@v4", workflow)
        self.assertLess(
            workflow.index("name: Zip wave checkpoint"),
            workflow.index("name: Deploy to GitHub Pages"),
        )
        self.assertIn("if: always()", workflow)
        self.assertIn("overwrite: true", workflow)
        self.assertIn("docs/_scripts/zip_jekyll_checkpoint.sh", workflow)
        self.assertIn("repository_dispatch:", workflow)
        self.assertIn("types: [jekyll-wave]", workflow)
        self.assertNotIn("workflow_run:", workflow)
        self.assertIn("actions/checkout@v4", workflow)
        self.assertIn("dawidd6/action-download-artifact@v2", workflow)
        self.assertIn("peaceiris/actions-gh-pages@v3", workflow)
        self.assertIn("ruby/setup-ruby@v1", workflow)
        self.assertIn("if-no-files-found: error", workflow)
        self.assertIn("gh api --method POST repos/${{ github.repository }}/dispatches", workflow)
        self.assertNotIn("peter-evans/repository-dispatch", workflow)
        self.assertNotIn("dawidd6/action-download-artifact@v6", workflow)
        self.assertNotIn("dawidd6/action-download-artifact@v21", workflow)
        self.assertNotIn("peaceiris/actions-gh-pages@v4", workflow)
        self.assertNotIn("peaceiris/actions-gh-pages@47f197a", workflow)
        self.assertNotIn("for wave in", workflow)
        self.assertLess(
            workflow.index("name: Upload wave checkpoint"),
            workflow.index("event_type=jekyll-wave"),
        )


class SearchIndexBatchTests(unittest.TestCase):
    def test_partial_wave_skips_s3_and_elasticsearch_prune(self):
        plugin = PLUGIN.read_text()
        self.assertIn("BatchLimiter.complete?", plugin)
        self.assertIn("upload_file_to_s3_bucket(filename) if BatchLimiter.complete?", plugin)
        prune = plugin.split(
            "delete_by_query index: ELASTICSEARCH_INDEX_NAME, body: {query: {bool: { must:",
            1,
        )[0]
        self.assertIn("models_json = backup_models_data.merge(models_json)", prune)
        self.assertIn(
            "BatchLimiter.complete?",
            plugin.split("must_not: {ids: {values: models_json.keys}}}", 1)[0],
        )

    def test_workflow_loops_waves_and_keeps_metadata_between_them(self):
        workflow = WORKFLOW.read_text()
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("JEKYLL_POST_BATCH", workflow)
        self.assertNotIn("rm -f .jekyll-metadata", workflow)
        self.assertIn("jekyll_wave_complete", workflow)
        self.assertIn("if: ${{ steps.jekyll-waves.outputs.jekyll_wave_complete == 'true' }}", workflow)
