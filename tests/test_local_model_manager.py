from __future__ import annotations

import hashlib
import json
from pathlib import Path

from spec_pipeline.local.model_manager import ModelCacheManager, ModelNotReadyError
from spec_pipeline.local.model_manifest import (
    LocalModelSpec,
    ModelFileSpec,
    ModelManifest,
    default_model_manifest,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _test_files() -> dict[str, bytes]:
    return {
        "config.json": b'{"model_type":"qwen3"}\n',
        "tokenizer.json": b'{"version":"test"}\n',
        "model.safetensors": b"fake weights\n",
    }


def _test_manifest(
    files: dict[str, bytes] | None = None, revision: str = "rev-111"
) -> ModelManifest:
    payload = files or _test_files()
    return ModelManifest(
        models=(
            LocalModelSpec(
                key="qwen3_8b_mlx",
                aliases=("qwen3-8b",),
                title="Qwen3 8B MLX 4-bit",
                hf_id="mlx-community/Qwen3-8B-4bit",
                revision=revision,
                backend="mlx",
                path="mlx/qwen3-8b",
                platforms=("macos",),
                disk_gb=1,
                files=tuple(
                    ModelFileSpec(path, len(data), _sha256(data)) for path, data in payload.items()
                ),
            ),
        )
    )


class FakeDownloader:
    def __init__(self, files: dict[str, bytes] | None = None, *, fail: bool = False) -> None:
        self.files = files or _test_files()
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def resolve_revision(self, *, repo_id: str, revision: str) -> str:
        return revision

    def snapshot_download(self, **kwargs) -> str:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("download failed")
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        for rel_path, data in self.files.items():
            path = local_dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return str(local_dir)


class FailingRevisionDownloader(FakeDownloader):
    def resolve_revision(self, *, repo_id: str, revision: str) -> str:
        raise RuntimeError("network unavailable")


class PreservePartialDownloader(FakeDownloader):
    def snapshot_download(self, **kwargs) -> str:
        local_dir = Path(str(kwargs["local_dir"]))
        assert (local_dir / "partial.bin").exists()
        return super().snapshot_download(**kwargs)


class SparseGgufDownloader(FakeDownloader):
    def __init__(self, *, expected_size: int) -> None:
        super().__init__({})
        self.expected_size = expected_size

    def snapshot_download(self, **kwargs) -> str:
        self.calls.append(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        model_path = local_dir / "model.gguf"
        assert not model_path.exists()
        with model_path.open("wb") as handle:
            handle.truncate(self.expected_size)
        return str(local_dir)


class NetworkForbidden:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def resolve_revision(self, *, repo_id: str, revision: str) -> str:
        return revision

    def snapshot_download(self, **kwargs) -> str:
        self.calls.append(kwargs)
        raise AssertionError("network/download should not be used")


def _install_ready_model(cache_dir: Path) -> None:
    files = _test_files()
    model_dir = cache_dir / "mlx" / "qwen3-8b"
    model_dir.mkdir(parents=True)
    for rel_path, data in files.items():
        (model_dir / rel_path).write_bytes(data)
    lock = {
        "version": 1,
        "models": {
            "qwen3_8b_mlx": {
                "hf_id": "mlx-community/Qwen3-8B-4bit",
                "revision": "rev-111",
                "path": "mlx/qwen3-8b",
                "status": "ready",
                "downloaded_at": "2026-04-26T12:00:00Z",
                "size_bytes": sum(len(data) for data in files.values()),
                "files_checked": True,
            }
        },
    }
    (cache_dir / "manifest.lock.json").write_text(
        json.dumps(lock, ensure_ascii=False), encoding="utf-8"
    )


def test_download_skips_network_when_model_is_ready(tmp_path: Path) -> None:
    _install_ready_model(tmp_path)
    downloader = NetworkForbidden()
    manager = ModelCacheManager(
        cache_dir=tmp_path, manifest=_test_manifest(), downloader=downloader
    )

    result = manager.download("qwen3-8b")

    assert result.ok is True
    assert result.status == "already_installed"
    assert downloader.calls == []
    lock = json.loads((tmp_path / "manifest.lock.json").read_text(encoding="utf-8"))
    assert lock["models"]["qwen3_8b_mlx"]["revision"] == "rev-111"


def test_download_writes_ready_lock_after_successful_verify(tmp_path: Path) -> None:
    downloader = FakeDownloader()
    manager = ModelCacheManager(
        cache_dir=tmp_path, manifest=_test_manifest(), downloader=downloader
    )

    result = manager.download("qwen3-8b")

    assert result.ok is True
    assert result.status == "ready"
    assert downloader.calls[0]["repo_id"] == "mlx-community/Qwen3-8B-4bit"
    assert downloader.calls[0]["revision"] == "rev-111"
    assert downloader.calls[0]["resume_download"] is True
    lock = json.loads((tmp_path / "manifest.lock.json").read_text(encoding="utf-8"))
    assert lock["models"]["qwen3_8b_mlx"]["status"] == "ready"
    assert lock["models"]["qwen3_8b_mlx"]["files_checked"] is True


def test_download_uses_manifest_download_patterns(tmp_path: Path) -> None:
    files = {"model.gguf": b"fake gguf", "README.md": b"readme"}
    manifest = ModelManifest(
        models=(
            LocalModelSpec(
                key="qwen3_6_safe",
                aliases=("qwen3.6-safe",),
                title="Qwen3.6 Safe",
                hf_id="bartowski/Qwen_Qwen3.6-27B-GGUF",
                revision="rev-444",
                backend="llama-cpp",
                path="gguf/qwen3.6-safe",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(
                    ModelFileSpec(
                        "model.gguf", len(files["model.gguf"]), _sha256(files["model.gguf"])
                    ),
                ),
                download_patterns=("model.gguf", "README.md"),
            ),
        )
    )
    downloader = FakeDownloader(files)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=manifest, downloader=downloader)

    result = manager.download("qwen3.6-safe")

    assert result.ok is True
    assert downloader.calls[0]["allow_patterns"] == ("model.gguf", "README.md")


def test_download_preserves_partial_files_for_resume(tmp_path: Path) -> None:
    partial_dir = tmp_path / "mlx" / "qwen3-8b"
    partial_dir.mkdir(parents=True)
    (partial_dir / "partial.bin").write_bytes(b"partial")
    downloader = PreservePartialDownloader()
    manager = ModelCacheManager(
        cache_dir=tmp_path, manifest=_test_manifest(), downloader=downloader
    )

    result = manager.download("qwen3-8b")

    assert result.ok is True
    assert downloader.calls[0]["resume_download"] is True


def test_download_resumes_partial_main_using_locked_revision(tmp_path: Path) -> None:
    partial_dir = tmp_path / "mlx" / "qwen3-8b"
    partial_dir.mkdir(parents=True)
    (partial_dir / "partial.bin").write_bytes(b"partial")
    (tmp_path / "manifest.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "qwen3_8b_mlx": {
                        "hf_id": "mlx-community/Qwen3-8B-4bit",
                        "revision": "rev-locked",
                        "path": "mlx/qwen3-8b",
                        "status": "partial",
                        "downloaded_at": "2026-04-26T12:00:00Z",
                        "size_bytes": 7,
                        "files_checked": False,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    downloader = PreservePartialDownloader()
    manager = ModelCacheManager(
        cache_dir=tmp_path, manifest=_test_manifest(revision="main"), downloader=downloader
    )

    result = manager.download("qwen3-8b")

    assert result.ok is True
    assert downloader.calls[0]["revision"] == "rev-locked"


def test_download_removes_tiny_corrupt_gguf_before_retry(tmp_path: Path) -> None:
    expected_size = 128 * 1024 * 1024
    manifest = ModelManifest(
        models=(
            LocalModelSpec(
                key="qwen_safe",
                aliases=("qwen-safe",),
                title="Qwen Safe",
                hf_id="example/qwen",
                revision="rev-gguf",
                backend="llama-cpp",
                path="gguf/qwen-safe",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(ModelFileSpec("model.gguf", expected_size),),
                download_patterns=("model.gguf",),
            ),
        )
    )
    model_dir = tmp_path / "gguf" / "qwen-safe"
    model_dir.mkdir(parents=True)
    (model_dir / "model.gguf").write_bytes(b"<html>blocked</html>")
    (tmp_path / "manifest.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "qwen_safe": {
                        "hf_id": "example/qwen",
                        "revision": "rev-gguf",
                        "path": "gguf/qwen-safe",
                        "status": "corrupt",
                        "downloaded_at": "2026-06-07T00:00:00Z",
                        "size_bytes": 20,
                        "files_checked": False,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    downloader = SparseGgufDownloader(expected_size=expected_size)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=manifest, downloader=downloader)

    result = manager.download("qwen-safe")

    assert result.ok is True
    assert result.status == "ready"
    assert (model_dir / "model.gguf").stat().st_size == expected_size
    assert downloader.calls[0]["revision"] == "rev-gguf"


def test_download_accepts_complete_gguf_larger_than_recorded_size(tmp_path: Path) -> None:
    # Regression: a finished GGUF whose real size is a few % ABOVE the manifest's
    # recorded size must validate as ready, not be rejected as "corrupt". HuggingFace
    # re-uploads / re-quantizations shift the byte count; the customer saw
    # "size mismatch: got 12.0 GB, expected 11.5 GB" on a fully-downloaded model and
    # the install failed even though the file was complete.
    recorded_size = 128 * 1024 * 1024
    actual_size = recorded_size + recorded_size // 20  # +5%, complete file
    manifest = ModelManifest(
        models=(
            LocalModelSpec(
                key="qwen_safe",
                aliases=("qwen-safe",),
                title="Qwen Safe",
                hf_id="example/qwen",
                revision="rev-gguf",
                backend="llama-cpp",
                path="gguf/qwen-safe",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(ModelFileSpec("model.gguf", recorded_size),),
                download_patterns=("model.gguf",),
            ),
        )
    )
    downloader = SparseGgufDownloader(expected_size=actual_size)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=manifest, downloader=downloader)

    result = manager.download("qwen-safe")

    assert result.ok is True
    assert result.status == "ready"
    assert (tmp_path / "gguf" / "qwen-safe" / "model.gguf").stat().st_size == actual_size


def test_download_corrupt_tiny_gguf_reports_retry_hint_and_removes_file(tmp_path: Path) -> None:
    expected_size = 128 * 1024 * 1024
    manifest = ModelManifest(
        models=(
            LocalModelSpec(
                key="qwen_safe",
                aliases=("qwen-safe",),
                title="Qwen Safe",
                hf_id="example/qwen",
                revision="rev-gguf",
                backend="llama-cpp",
                path="gguf/qwen-safe",
                platforms=("windows-wsl",),
                disk_gb=1,
                files=(ModelFileSpec("model.gguf", expected_size),),
                download_patterns=("model.gguf",),
            ),
        )
    )
    downloader = FakeDownloader({"model.gguf": b"<html>blocked</html>"})
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=manifest, downloader=downloader)

    result = manager.download("qwen-safe")

    assert result.ok is False
    assert result.status == "corrupt"
    assert "got" in result.model.issues[0]
    assert "expected 128.0 MB" in result.model.issues[0]
    assert any(
        issue.startswith("removed tiny invalid file: model.gguf") for issue in result.model.issues
    )
    assert "HuggingFace/VPN/proxy" in result.message
    assert not (tmp_path / "gguf" / "qwen-safe" / "model.gguf").exists()


def test_download_handles_revision_resolution_failure_without_traceback(tmp_path: Path) -> None:
    manifest = _test_manifest(revision="main")
    manager = ModelCacheManager(
        cache_dir=tmp_path, manifest=manifest, downloader=FailingRevisionDownloader()
    )

    result = manager.download("qwen3-8b")

    assert result.ok is False
    assert result.status == "failed"
    assert "network unavailable" in result.message


def test_corrupt_lock_file_raises_cache_lock_error(tmp_path: Path) -> None:
    (tmp_path / "manifest.lock.json").write_text("{bad", encoding="utf-8")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_test_manifest())

    try:
        manager.list_statuses()
    except Exception as exc:
        assert exc.__class__.__name__ == "ModelCacheLockError"
    else:
        raise AssertionError("corrupt lock file must not be ignored")


def test_verify_marks_corrupt_when_checksum_mismatches(tmp_path: Path) -> None:
    _install_ready_model(tmp_path)
    (tmp_path / "mlx" / "qwen3-8b" / "model.safetensors").write_bytes(b"bad")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_test_manifest())

    result = manager.verify("qwen3-8b")

    assert result.ok is False
    assert result.status == "corrupt"
    assert "checksum mismatch: model.safetensors" in result.model.issues
    lock = json.loads((tmp_path / "manifest.lock.json").read_text(encoding="utf-8"))
    assert lock["models"]["qwen3_8b_mlx"]["status"] == "corrupt"


def test_repair_redownloads_corrupt_model_with_pinned_revision(tmp_path: Path) -> None:
    _install_ready_model(tmp_path)
    (tmp_path / "mlx" / "qwen3-8b" / "model.safetensors").write_bytes(b"bad")
    downloader = FakeDownloader()
    manager = ModelCacheManager(
        cache_dir=tmp_path, manifest=_test_manifest(), downloader=downloader
    )

    result = manager.repair("qwen3-8b")

    assert result.ok is True
    assert downloader.calls[0]["revision"] == "rev-111"
    assert manager.status("qwen3-8b").status == "ready"


def test_update_requires_confirm_for_ready_model(tmp_path: Path) -> None:
    _install_ready_model(tmp_path)
    downloader = NetworkForbidden()
    manager = ModelCacheManager(
        cache_dir=tmp_path,
        manifest=_test_manifest(revision="rev-222"),
        downloader=downloader,
    )

    result = manager.update("qwen3-8b", confirm=False)

    assert result.ok is False
    assert result.status == "confirmation_required"
    assert downloader.calls == []


def test_remove_requires_confirm_for_ready_model(tmp_path: Path) -> None:
    _install_ready_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_test_manifest())

    result = manager.remove("qwen3-8b", confirm=False)

    assert result.ok is False
    assert result.status == "confirmation_required"
    assert (tmp_path / "mlx" / "qwen3-8b").exists()


def test_remove_with_confirm_deletes_model_and_lock(tmp_path: Path) -> None:
    _install_ready_model(tmp_path)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_test_manifest())

    result = manager.remove("qwen3-8b", confirm=True)

    assert result.ok is True
    assert not (tmp_path / "mlx" / "qwen3-8b").exists()
    assert manager.status("qwen3-8b").status == "missing"


def test_resolve_for_run_never_downloads_and_requires_ready_model(tmp_path: Path) -> None:
    manager = ModelCacheManager(
        cache_dir=tmp_path, manifest=_test_manifest(), downloader=NetworkForbidden()
    )

    try:
        manager.resolve_for_run("qwen3-8b")
    except ModelNotReadyError as exc:
        assert "not ready" in str(exc)
    else:
        raise AssertionError("missing model must not resolve for run")


def test_default_manifest_rejects_config_only_partial_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "mlx" / "mlx-community--Qwen3-8B-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "qwen3_8b_mlx": {
                        "hf_id": "mlx-community/Qwen3-8B-4bit",
                        "revision": "main",
                        "path": "mlx/mlx-community--Qwen3-8B-4bit",
                        "status": "ready",
                        "downloaded_at": "2026-04-26T12:00:00Z",
                        "size_bytes": 2,
                        "files_checked": True,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=default_model_manifest())

    result = manager.verify("qwen3-8b")

    assert result.ok is False
    assert "missing file pattern: *.safetensors" in result.model.issues


def test_default_manifest_rejects_qwen_without_tokenizer(tmp_path: Path) -> None:
    model_dir = tmp_path / "mlx" / "mlx-community--Qwen3-8B-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=default_model_manifest())

    result = manager.verify("qwen3-8b")

    assert result.ok is False
    assert "missing file: tokenizer.json" in result.model.issues


def test_default_manifest_includes_qwen36_mlx_vlm() -> None:
    spec = default_model_manifest().resolve("qwen3.6-35b-a3b")

    assert spec.backend == "mlx-vlm"
    assert spec.hf_id == "mlx-community/Qwen3.6-35B-A3B-4bit"
    assert spec.path == "mlx-vlm/mlx-community--Qwen3.6-35B-A3B-4bit"
    assert spec.min_ram_gb == 64


def test_default_manifest_includes_qwen36_windows_gguf_profiles() -> None:
    manifest = default_model_manifest()
    safe = manifest.resolve("qwen3.6-safe")
    balanced = manifest.resolve("qwen3.6-balanced")
    quality = manifest.resolve("qwen3.6-quality")

    assert safe.backend == "llama-cpp"
    assert safe.files[0].path.endswith("IQ3_XXS.gguf")
    assert safe.download_patterns == ("Qwen_Qwen3.6-27B-IQ3_XXS.gguf", "README.md")
    assert manifest.resolve("qwen36-safe").key == "qwen3_6_27b_gguf_iq3_xxs"
    assert manifest.resolve("qwen36-balanced").key == "qwen3_6_27b_gguf_q4_k_s"
    assert manifest.resolve("qwen36-quality").key == "qwen3_6_27b_gguf_q4_k_m"
    assert balanced.files[0].path.endswith("Q4_K_S.gguf")
    assert quality.files[0].path.endswith("Q4_K_M.gguf")


def test_default_manifest_uses_qwen36_max_alias_for_gptq_profile() -> None:
    manifest = default_model_manifest()
    spec = manifest.resolve("qwen36-max")

    assert spec.key == "qwen3_6_27b_gptq_cuda"
    assert spec.backend == "vllm"
    assert manifest.resolve("qwen36-max-vllm").key == "qwen3_6_27b_gptq_cuda"
    assert manifest.resolve("qwen3.6-vllm-experimental").key == "qwen3_6_27b_gptq_cuda"
    assert "qwen3.6-" + "27b-gptq" not in spec.aliases


def test_default_manifest_rejects_qwen36_vlm_without_processor_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "mlx-vlm" / "mlx-community--Qwen3.6-35B-A3B-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=default_model_manifest())

    result = manager.verify("qwen3.6-35b-a3b")

    assert result.ok is False
    assert "missing file: processor_config.json" in result.model.issues
    assert "missing file: preprocessor_config.json" in result.model.issues


def test_status_requires_ready_lock_entry_even_when_files_verify(tmp_path: Path) -> None:
    files = _test_files()
    model_dir = tmp_path / "mlx" / "qwen3-8b"
    model_dir.mkdir(parents=True)
    for rel_path, data in files.items():
        (model_dir / rel_path).write_bytes(data)
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=_test_manifest())

    status = manager.status("qwen3-8b")

    assert status.status == "untracked"
    assert status.installed is False


def test_verify_rejects_file_pattern_traversal(tmp_path: Path) -> None:
    (tmp_path / "outside.safetensors").write_bytes(b"outside")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    manifest = ModelManifest(
        models=(
            LocalModelSpec(
                key="unsafe",
                aliases=("unsafe",),
                title="Unsafe",
                hf_id="example/unsafe",
                revision="rev-1",
                backend="test",
                path="model",
                platforms=("test",),
                disk_gb=1,
                files=(ModelFileSpec("config.json"),),
                file_patterns=("../outside.safetensors",),
            ),
        )
    )
    manager = ModelCacheManager(cache_dir=tmp_path, manifest=manifest)

    result = manager.verify("unsafe")

    assert result.ok is False
    assert "invalid file pattern: ../outside.safetensors" in result.model.issues
