#!/usr/bin/env python3
"""Fetch a reviewed upstream build; seed native modules before importing AITER.

Build-time only. Never run seed against a live installation. No Triton/FlyDSL
compiler caches or upstream Python sources are transplanted into the checkout.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY = "ROCm/aiter"
API = f"https://api.github.com/repos/{REPOSITORY}"
# Reviewed scheduled producer: pinned submodules, CPython 3.10, gfx942/gfx950,
# rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1.
# Dispatch runs can override the image and are deliberately ineligible.
RECIPES = {
    "nightly.yaml": "35c728f5c2aa81b33aaf5dec8603d3b01a92f2ecaef129d363532e123461d974",
    "aiter-release.yaml": "056f9671ccd547b069f70c093c73b7f5675977855c6a1c0a7a0e72cc654d5605",
}
# The Dockerfile's native backport affects only this module at the reviewed pin.
# Unknown changes (including shared headers, recipes and submodules) drop ALL
# prebuilt modules rather than guessing their dependency closure.
REVIEWED_COMMIT = "456b92780c8b650c1e3e4b0fa1ca21f0d1fb363d"
NATIVE_PATCHES = {
    "csrc/kernels/dsv4_rotate_quant.cu": "module_dsv4_rotate_quant",
    # Fork-local GLM kernel. It is absent from the upstream wheel, so the
    # exclusion is a no-op and only this module compiles on first use.
    "csrc/glm_prefill_stage1_partials.cu": "module_glm_prefill_stage1_partials_v1",
}
PYTHON_PATCHES = {
    "aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py",
    "csrc/cpp_itfs/torch_utils.py",
    # Fork-local GLM-5.3-Flash prefill. All three are Python-only or a test
    # module; no wheel module is compiled from them, so reuse stays valid.
    "aiter/fused_moe.py",
    "aiter/ops/glm_prefill_stage1_partials.py",
    "op_tests/probe_glm_prefill_stage1.py",
}


class CacheMiss(Exception):
    """No artifact proven compatible; leave compilation to the source build."""


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise ValueError("Refusing non-HTTPS artifact redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            # GitHub redirects artifacts to signed blob URLs. Never forward the
            # repository's GITHUB_TOKEN to object storage or another origin.
            redirected.remove_header("Authorization")
        return redirected


def open_url(url):
    headers = {"User-Agent": "furnace-kernel-build"}
    if url.startswith("https://api.github.com/"):
        headers["Accept"] = "application/vnd.github+json"
        if os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return urllib.request.build_opener(SafeRedirect()).open(
        urllib.request.Request(url, headers=headers), timeout=60
    )


def get_json(url):
    with open_url(url) as response:
        return json.load(response)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(commit, destination):
    destination.mkdir(parents=True, exist_ok=True)
    # Clear only this fetcher's outputs, so a miss cannot reuse yesterday's hit.
    for old in destination.glob("*.whl"):
        old.unlink()
    manifest = {"available": False, "commit": commit}
    (destination / "manifest.json").write_text(json.dumps(manifest) + "\n")
    try:
        if commit != REVIEWED_COMMIT:
            raise CacheMiss("AITER revision has not been reviewed for binary reuse")
        for recipe, expected in RECIPES.items():
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/.github/workflows/{recipe}"
            with open_url(url) as response:
                actual = hashlib.sha256(response.read()).hexdigest()
            if actual != expected:
                raise CacheMiss(f"Unreviewed upstream build recipe: {recipe}")
        runs = get_json(
            f"{API}/actions/workflows/nightly.yaml/runs?head_sha={commit}&status=success&event=schedule&per_page=100"
        )["workflow_runs"]
        artifact = None
        for run in runs:
            if not (
                run["head_sha"] == commit
                and run["event"] == "schedule"
                and run["conclusion"] == "success"
                and run["head_repository"]["full_name"] == REPOSITORY
                and run["path"] == ".github/workflows/nightly.yaml"
            ):
                continue
            artifacts = get_json(
                f"{API}/actions/runs/{run['id']}/artifacts?per_page=100"
            )["artifacts"]
            matches = [
                a
                for a in artifacts
                if not a["expired"]
                and a["name"]
                == f"aiter-whl-packages-py3.10-{run['id']}-{run['run_attempt']}"
                and a.get("workflow_run", {}).get("head_sha") == commit
            ]
            if len(matches) == 1:
                artifact = matches[0]
                break
        if artifact is None:
            raise CacheMiss(
                "No successful matching Torch 2.9.1 / ROCm 7.2 nightly artifact"
            )
        expected = artifact.get("digest", "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
            raise CacheMiss("Upstream artifact has no SHA256 digest")
        with tempfile.TemporaryDirectory(prefix="furnace-aiter-") as temporary:
            archive = Path(temporary) / "artifact.zip"
            with (
                open_url(f"{API}/actions/artifacts/{artifact['id']}/zip") as response,
                archive.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            if digest_file(archive) != expected.removeprefix("sha256:"):
                raise ValueError("Upstream artifact SHA256 mismatch")
            with zipfile.ZipFile(archive) as package:
                wheels = [
                    n
                    for n in package.namelist()
                    if re.fullmatch(r"amd_aiter-[^/]+-cp310-cp310-linux_x86_64\.whl", n)
                ]
                if len(wheels) != 1:
                    raise ValueError("Expected exactly one CPython 3.10 Linux wheel")
                wheel = destination / wheels[0]
                with package.open(wheels[0]) as source, wheel.open("wb") as output:
                    shutil.copyfileobj(source, output)
            manifest.update(
                available=True,
                repository=REPOSITORY,
                wheel=wheel.name,
                sha256=digest_file(wheel),
                artifact_id=artifact["id"],
                artifact_digest=expected,
                run_id=run["id"],
                recipes=RECIPES,
                python="3.10",
                torch="2.9.1",
                rocm="7.2",
                gpu_arch="gfx942-rocm720",
            )
    except (CacheMiss, urllib.error.URLError, TimeoutError) as error:
        manifest["reason"] = str(error)
        print(f"AITER prebuilt miss: {error}; using source/JIT (no build-all).")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["available"]:
        print(
            f"Downloaded verified AITER artifact {manifest['artifact_id']}: {manifest['wheel']}"
        )


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def excluded_modules(source, commit):
    if git(source, "rev-parse", "HEAD") != commit or commit != REVIEWED_COMMIT:
        raise CacheMiss("Source revision differs from the reviewed artifact")
    changed = set(
        git(
            source, "diff", "--name-only", "--ignore-submodules=none", "HEAD"
        ).splitlines()
    )
    changed.update(
        git(source, "ls-files", "--others", "--exclude-standard").splitlines()
    )
    unknown = changed - PYTHON_PATCHES - NATIVE_PATCHES.keys()
    if unknown:
        raise CacheMiss(
            f"Unreviewed source changes require compilation: {sorted(unknown)}"
        )
    return {NATIVE_PATCHES[p] for p in changed if p in NATIVE_PATCHES}


def seed(commit, directory, source, gpu_arch):
    # Source edits must never leave a name-addressed AITER cache active. This
    # command belongs before setup/import, in a fresh Docker build process.
    jit = source / "aiter/jit"
    for old in jit.glob("*.so"):
        old.unlink()
    shutil.rmtree(jit / "build", ignore_errors=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("available"):
        raise CacheMiss(manifest.get("reason", "No prebuilt input supplied"))
    if (
        manifest.get("commit") != commit
        or manifest.get("repository") != REPOSITORY
        or manifest.get("recipes") != RECIPES
        or gpu_arch != manifest.get("gpu_arch")
        or gpu_arch != "gfx942-rocm720"
    ):
        raise CacheMiss("Prebuilt source, producer or GPU flavor mismatch")
    import torch

    if (
        sys.version_info[:2] != (3, 10)
        or torch.__version__.split("+")[0] != "2.9.1"
        or not torch.version.hip
        or not torch.version.hip.startswith("7.2.")
    ):
        raise CacheMiss("Prebuilt Python / Torch / ROCm ABI mismatch")
    excluded = excluded_modules(source, commit)
    wheel_name = manifest["wheel"]
    if Path(wheel_name).name != wheel_name:
        raise ValueError("Invalid wheel filename")
    wheel = directory / wheel_name
    if digest_file(wheel) != manifest["sha256"]:
        raise ValueError("Prebuilt wheel SHA256 mismatch")
    with zipfile.ZipFile(wheel) as package:
        modules = [
            n for n in package.namelist() if re.fullmatch(r"aiter/jit/[^/]+\.so", n)
        ]
        if "aiter/jit/module_aiter_core.so" not in modules:
            raise ValueError("Prebuilt wheel has no AITER native core")
        count = 0
        for name in modules:
            basename = Path(name).name
            if any(
                basename.startswith(module + ".") or basename.startswith(module + "_")
                for module in excluded
            ):
                continue
            with package.open(name) as origin, (jit / basename).open("wb") as output:
                shutil.copyfileobj(origin, output)
            count += 1
    print(f"Reused {count} AITER HIP modules; rebuild from source: {sorted(excluded)}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "seed"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--gpu-arch")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit must be a full Git SHA")
    if args.mode == "fetch":
        fetch(args.commit, args.directory)
    else:
        if args.source is None or args.gpu_arch is None:
            parser.error("seed requires --source and --gpu-arch")
        try:
            seed(args.commit, args.directory, args.source, args.gpu_arch)
        except CacheMiss as error:
            print(f"AITER prebuilt miss: {error}; compiling from source.")
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
