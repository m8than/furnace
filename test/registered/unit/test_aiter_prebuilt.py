import importlib.util
import json
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "aiter_prebuilt", Path(__file__).resolve().parents[3] / "docker/aiter_prebuilt.py"
)
prebuilt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prebuilt)


class TestAiterPrebuilt(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.native = self.source / "csrc/kernels/dsv4_rotate_quant.cu"
        self.native.parent.mkdir(parents=True)
        self.native.write_text("int version = 1;\n")
        self.header = self.source / "csrc/common.h"
        self.header.write_text("#define VERSION 1\n")
        self.run_git("init", "-q")
        self.run_git("add", ".")
        self.run_git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "baseline",
        )
        self.commit = self.run_git("rev-parse", "HEAD").strip()
        self.reviewed = patch.object(prebuilt, "REVIEWED_COMMIT", self.commit)
        self.reviewed.start()
        self.addCleanup(self.reviewed.stop)

    def run_git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.source), *args], text=True
        )

    def test_native_backport_excludes_only_its_module(self):
        self.native.write_text("int version = 2;\n")
        self.run_git("add", ".")  # Docker's cherry-pick is staged, not committed.
        self.assertEqual(
            prebuilt.excluded_modules(self.source, self.commit),
            {"module_dsv4_rotate_quant"},
        )

    def test_shared_header_change_rejects_prebuilt_modules(self):
        self.header.write_text("#define VERSION 2\n")
        with self.assertRaises(prebuilt.CacheMiss):
            prebuilt.excluded_modules(self.source, self.commit)

    def test_untracked_codegen_input_rejects_prebuilt_modules(self):
        (self.native.parent / "new_kernel.cu").write_text("int new_kernel;\n")
        with self.assertRaises(prebuilt.CacheMiss):
            prebuilt.excluded_modules(self.source, self.commit)

    def test_glm_native_patch_excludes_only_its_module(self):
        glm = self.source / "csrc/glm_prefill_stage1_partials.cu"
        glm.write_text("int glm_version = 1;\n")
        self.run_git("add", ".")
        self.assertEqual(
            prebuilt.excluded_modules(self.source, self.commit),
            {"module_glm_prefill_stage1_partials_v1"},
        )

    def test_shipped_glm_patches_stay_inside_the_allow_lists(self):
        # A patched file missing from the allow lists makes excluded_modules
        # raise CacheMiss, which silently drops every prebuilt module and
        # forces a source build. Fail here instead.
        directory = Path(prebuilt.__file__).resolve().parent / "aiter-glmp-patches"
        patches = sorted(directory.glob("*.patch"))
        self.assertTrue(patches, f"no fork GLM patches under {directory}")
        known = prebuilt.PYTHON_PATCHES | set(prebuilt.NATIVE_PATCHES)
        for patch_file in patches:
            listed = subprocess.check_output(
                ["git", "apply", "--numstat", str(patch_file)], text=True
            )
            paths = {
                line.split("\t")[-1] for line in listed.splitlines() if line.strip()
            }
            self.assertTrue(paths, f"{patch_file.name} declares no paths")
            self.assertLessEqual(
                paths, known, f"{patch_file.name} touches unlisted paths"
            )

    def test_missing_artifact_removes_old_native_cache(self):
        jit = self.source / "aiter/jit"
        (jit / "build/module_aiter_core").mkdir(parents=True)
        (jit / "module_aiter_core.so").write_bytes(b"stale core")
        (jit / "build/module_aiter_core/old.o").write_bytes(b"stale object")
        incoming = self.root / "incoming"
        incoming.mkdir()
        (incoming / "manifest.json").write_text('{"available": false}')
        with self.assertRaises(prebuilt.CacheMiss):
            prebuilt.seed(self.commit, incoming, self.source, "gfx942-rocm720")
        self.assertFalse((jit / "module_aiter_core.so").exists())
        self.assertFalse((jit / "build").exists())

    def test_failed_fetch_cannot_leave_previous_hit_active(self):
        incoming = self.root / "incoming"
        incoming.mkdir()
        (incoming / "old.whl").write_bytes(b"old artifact")
        (incoming / "manifest.json").write_text('{"available": true}')
        with patch.object(
            prebuilt, "open_url", side_effect=urllib.error.URLError("offline")
        ):
            prebuilt.fetch(self.commit, incoming)
        self.assertFalse((incoming / "old.whl").exists())
        self.assertFalse(
            json.loads((incoming / "manifest.json").read_text())["available"]
        )

    def test_signed_storage_redirect_does_not_receive_github_token(self):
        request = urllib.request.Request(
            "https://api.github.com/repos/ROCm/aiter/actions/artifacts/1/zip",
            headers={"Authorization": "Bearer secret"},
        )
        redirected = prebuilt.SafeRedirect().redirect_request(
            request, None, 302, "Found", {}, "https://storage.example.invalid/signed"
        )
        self.assertIsNone(redirected.get_header("Authorization"))
        with self.assertRaises(ValueError):
            prebuilt.SafeRedirect().redirect_request(
                request, None, 302, "Found", {}, "http://storage.example.invalid/signed"
            )


if __name__ == "__main__":
    unittest.main()
