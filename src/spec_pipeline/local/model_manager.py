from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from spec_pipeline.local.model_manifest import LocalModelSpec, ModelManifest, default_model_manifest

LOCK_FILE_NAME = "manifest.lock.json"


class ModelNotReadyError(RuntimeError):
    pass


class ModelCacheLockError(RuntimeError):
    pass


class SnapshotDownloader(Protocol):
    def snapshot_download(
        self,
        *,
        repo_id: str,
        revision: str,
        local_dir: str,
        resume_download: bool,
        local_files_only: bool = False,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> str: ...

    def resolve_revision(self, *, repo_id: str, revision: str) -> str: ...


@dataclass(slots=True)
class ModelStatus:
    key: str
    title: str
    alias: str
    hf_id: str
    revision: str
    backend: str
    path: str
    status: str
    installed: bool
    message: str
    issues: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "title": self.title,
            "alias": self.alias,
            "hf_id": self.hf_id,
            "revision": self.revision,
            "backend": self.backend,
            "path": self.path,
            "status": self.status,
            "installed": self.installed,
            "message": self.message,
            "issues": self.issues,
        }


@dataclass(slots=True)
class ModelOperationResult:
    ok: bool
    status: str
    message: str
    model: ModelStatus

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "status": self.status,
            "message": self.message,
            "model": self.model.to_dict(),
        }


class HuggingFaceSnapshotDownloader:
    def resolve_revision(self, *, repo_id: str, revision: str) -> str:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Install local dependencies first."
            ) from exc
        return str(HfApi().model_info(repo_id=repo_id, revision=revision).sha)

    def snapshot_download(
        self,
        *,
        repo_id: str,
        revision: str,
        local_dir: str,
        resume_download: bool,
        local_files_only: bool = False,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> str:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Install local dependencies first."
            ) from exc
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "revision": revision,
            "local_dir": local_dir,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
        }
        if allow_patterns:
            kwargs["allow_patterns"] = list(allow_patterns)
        return snapshot_download(**kwargs)


def default_model_cache_dir(project_root: Path) -> Path:
    return project_root / "artifacts" / "models"


class ModelCacheManager:
    def __init__(
        self,
        *,
        cache_dir: str | Path,
        manifest: ModelManifest | None = None,
        downloader: SnapshotDownloader | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.manifest = manifest or default_model_manifest()
        self.downloader = downloader or HuggingFaceSnapshotDownloader()

    @property
    def lock_path(self) -> Path:
        return self.cache_dir / LOCK_FILE_NAME

    def list_statuses(self) -> list[ModelStatus]:
        return [self.status(spec.key) for spec in self.manifest.models]

    def status(self, model: str) -> ModelStatus:
        spec = self.manifest.resolve(model)
        lock = self._read_lock()
        entry = self._lock_entry(lock, spec)
        revision = str(entry.get("revision") or spec.revision) if entry else spec.revision
        model_dir = self._model_dir(spec)
        issues = self._verify_files(spec, model_dir)
        alias = spec.aliases[0] if spec.aliases else spec.key
        if not entry and not model_dir.exists():
            return ModelStatus(
                key=spec.key,
                title=spec.title,
                alias=alias,
                hf_id=spec.hf_id,
                revision=revision,
                backend=spec.backend,
                path=str(model_dir),
                status="missing",
                installed=False,
                message="модель не установлена",
                issues=[],
            )
        if not entry:
            return ModelStatus(
                key=spec.key,
                title=spec.title,
                alias=alias,
                hf_id=spec.hf_id,
                revision=revision,
                backend=spec.backend,
                path=str(model_dir),
                status="untracked" if not issues else "partial",
                installed=False,
                message="модель найдена без lock-записи"
                if not issues
                else "модель найдена частично, но lock-записи нет",
                issues=issues,
            )
        if issues:
            status = "partial" if entry and entry.get("status") == "partial" else "corrupt"
            return ModelStatus(
                key=spec.key,
                title=spec.title,
                alias=alias,
                hf_id=spec.hf_id,
                revision=revision,
                backend=spec.backend,
                path=str(model_dir),
                status=status,
                installed=False,
                message="модель найдена, но не прошла проверку",
                issues=issues,
            )
        if entry.get("status") != "ready":
            return ModelStatus(
                key=spec.key,
                title=spec.title,
                alias=alias,
                hf_id=spec.hf_id,
                revision=revision,
                backend=spec.backend,
                path=str(model_dir),
                status=str(entry.get("status") or "partial"),
                installed=False,
                message="модель не отмечена как готовая в lock-файле",
                issues=[],
            )
        return ModelStatus(
            key=spec.key,
            title=spec.title,
            alias=alias,
            hf_id=spec.hf_id,
            revision=revision,
            backend=spec.backend,
            path=str(model_dir),
            status="ready",
            installed=True,
            message="модель установлена и готова",
            issues=[],
        )

    def download(self, model: str) -> ModelOperationResult:
        spec = self.manifest.resolve(model)
        current = self.status(spec.key)
        if current.status == "ready":
            return ModelOperationResult(
                ok=True,
                status="already_installed",
                message="модель уже установлена, скачивание не требуется",
                model=current,
            )
        return self._download_spec(spec, revision=current.revision, replace_existing=True)

    def verify(self, model: str) -> ModelOperationResult:
        spec = self.manifest.resolve(model)
        current = self.status(spec.key)
        lock = self._read_lock()
        if current.status == "ready":
            self._set_lock_entry(lock, spec, status="ready", revision=current.revision)
            self._write_lock(lock)
            return ModelOperationResult(True, "ready", "модель прошла проверку", current)
        if current.status != "missing":
            self._set_lock_entry(lock, spec, status="corrupt", revision=current.revision)
            self._write_lock(lock)
        return ModelOperationResult(False, current.status, current.message, current)

    def repair(self, model: str) -> ModelOperationResult:
        spec = self.manifest.resolve(model)
        current = self.status(spec.key)
        if current.status == "ready":
            return ModelOperationResult(True, "already_ready", "модель уже исправна", current)
        return self._download_spec(spec, revision=current.revision, replace_existing=True)

    def update(self, model: str, *, confirm: bool = False) -> ModelOperationResult:
        spec = self.manifest.resolve(model)
        current = self.status(spec.key)
        if not confirm:
            return ModelOperationResult(
                False,
                "confirmation_required",
                "обновление модели требует явного --confirm",
                current,
            )
        return self._download_spec(spec, revision=spec.revision, replace_existing=False)

    def remove(self, model: str, *, confirm: bool = False) -> ModelOperationResult:
        spec = self.manifest.resolve(model)
        current = self.status(spec.key)
        if current.status != "missing" and not confirm:
            return ModelOperationResult(
                False,
                "confirmation_required",
                "удаление модели требует явного --confirm",
                current,
            )
        model_dir = self._model_dir(spec)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        lock = self._read_lock()
        lock.setdefault("models", {}).pop(spec.key, None)
        self._write_lock(lock)
        return ModelOperationResult(
            True,
            "removed",
            "модель удалена" if current.status != "missing" else "модель уже отсутствует",
            self.status(spec.key),
        )

    def resolve_for_run(self, model: str) -> Path:
        current = self.status(model)
        if current.status != "ready":
            raise ModelNotReadyError(
                f"Model '{model}' is not ready ({current.status}). Run models download/repair first."
            )
        return Path(current.path)

    def _download_spec(
        self, spec: LocalModelSpec, *, revision: str, replace_existing: bool
    ) -> ModelOperationResult:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self._model_dir(spec)
        previous_status = self.status(spec.key).status
        previous_lock = self._read_lock()

        try:
            resolved_revision = self._resolve_revision(spec, revision)
        except Exception as exc:
            return ModelOperationResult(
                False,
                "failed",
                f"ошибка подготовки загрузки: {exc}",
                self.status(spec.key),
            )

        download_dir = target_dir
        if not replace_existing:
            safe_revision = resolved_revision.replace("/", "_")
            download_dir = self.cache_dir / "downloads" / "partial" / f"{spec.key}-{safe_revision}"
            if download_dir.exists():
                shutil.rmtree(download_dir)

        lock = json.loads(json.dumps(previous_lock))
        should_record_partial = replace_existing or previous_status != "ready"
        if should_record_partial:
            self._set_lock_entry(lock, spec, status="partial", revision=resolved_revision)
            self._write_lock(lock)
        self._remove_suspicious_tiny_files(spec, download_dir)
        try:
            download_dir.mkdir(parents=True, exist_ok=True)
            self.downloader.snapshot_download(
                repo_id=spec.hf_id,
                revision=resolved_revision,
                local_dir=str(download_dir),
                resume_download=True,
                local_files_only=False,
                allow_patterns=spec.download_patterns or None,
            )
        except Exception as exc:
            if not should_record_partial:
                self._write_lock(previous_lock)
            failed = self.status(spec.key)
            return ModelOperationResult(False, "failed", f"ошибка загрузки: {exc}", failed)

        issues = self._verify_files(spec, download_dir)
        if issues:
            removed_files = self._remove_suspicious_tiny_files(spec, download_dir)
            if removed_files:
                issues.extend(
                    f"removed tiny invalid file: {path}; retry download after checking HuggingFace/VPN/proxy"
                    for path in removed_files
                )
            if should_record_partial:
                self._set_lock_entry(lock, spec, status="corrupt", revision=resolved_revision)
                self._write_lock(lock)
            else:
                self._write_lock(previous_lock)
            corrupt = ModelStatus(
                key=spec.key,
                title=spec.title,
                alias=spec.aliases[0] if spec.aliases else spec.key,
                hf_id=spec.hf_id,
                revision=resolved_revision,
                backend=spec.backend,
                path=str(download_dir),
                status="corrupt",
                installed=False,
                message=(
                    "модель скачана, но не прошла проверку. "
                    "Если файл оказался слишком маленьким, это обычно ошибка загрузки "
                    "HuggingFace/VPN/proxy; повторите установку после проверки сети."
                ),
                issues=issues,
            )
            return ModelOperationResult(False, "corrupt", corrupt.message, corrupt)
        if download_dir != target_dir:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(download_dir), str(target_dir))
        self._set_lock_entry(lock, spec, status="ready", revision=resolved_revision)
        self._write_lock(lock)
        return ModelOperationResult(
            True, "ready", "модель скачана и проверена", self.status(spec.key)
        )

    def _resolve_revision(self, spec: LocalModelSpec, revision: str) -> str:
        if revision != "main":
            return revision
        resolver = getattr(self.downloader, "resolve_revision", None)
        if callable(resolver):
            return str(resolver(repo_id=spec.hf_id, revision=revision))
        return revision

    def _model_dir(self, spec: LocalModelSpec) -> Path:
        path = (self.cache_dir / spec.path).resolve()
        cache_root = self.cache_dir.resolve()
        if cache_root not in path.parents and path != cache_root:
            raise ValueError(f"Model path escapes cache root: {spec.path}")
        return path

    def _verify_files(self, spec: LocalModelSpec, model_dir: Path) -> list[str]:
        if not model_dir.exists():
            return ["model directory is missing"]
        if not spec.files:
            if any(path.is_file() for path in model_dir.rglob("*")):
                return []
            return ["model directory is empty"]

        issues: list[str] = []
        for expected_file in spec.files:
            path = (model_dir / expected_file.path).resolve()
            model_root = model_dir.resolve()
            if model_root not in path.parents and path != model_root:
                issues.append(f"invalid file path: {expected_file.path}")
                continue
            if not path.exists():
                issues.append(f"missing file: {expected_file.path}")
                continue
            if not path.is_file():
                issues.append(f"not a regular file: {expected_file.path}")
                continue
            # The manifest size is approximate: HuggingFace re-uploads / re-quantizations
            # can shift a GGUF's byte count by a few percent, so an exact-match check
            # falsely rejected complete downloads that were merely a little larger than
            # recorded (customer saw "got 12.0 GB, expected 11.5 GB" on a finished file).
            # Only flag a clearly incomplete (truncated/blocked) download — i.e. well
            # below the recorded size — and accept anything at or above ~90% of it.
            if (
                expected_file.size_bytes is not None
                and path.stat().st_size * 10 < expected_file.size_bytes * 9
            ):
                issues.append(
                    "size mismatch: "
                    f"{expected_file.path} "
                    f"(got {format_bytes(path.stat().st_size)}, "
                    f"expected {format_bytes(expected_file.size_bytes)})"
                )
            if expected_file.sha256 is not None and sha256_file(path) != expected_file.sha256:
                issues.append(f"checksum mismatch: {expected_file.path}")
        for pattern in spec.file_patterns:
            pattern_path = Path(pattern)
            if pattern_path.is_absolute() or ".." in pattern_path.parts:
                issues.append(f"invalid file pattern: {pattern}")
                continue
            if not any(path.is_file() for path in model_dir.glob(pattern)):
                issues.append(f"missing file pattern: {pattern}")
        return issues

    def _remove_suspicious_tiny_files(self, spec: LocalModelSpec, model_dir: Path) -> list[str]:
        removed: list[str] = []
        if not model_dir.exists():
            return removed
        for expected_file in spec.files:
            if expected_file.size_bytes is None or expected_file.size_bytes < 100 * 1024 * 1024:
                continue
            path = (model_dir / expected_file.path).resolve()
            model_root = model_dir.resolve()
            if model_root not in path.parents and path != model_root:
                continue
            if not path.is_file():
                continue
            actual_size = path.stat().st_size
            if actual_size < min(1024 * 1024, expected_file.size_bytes // 1000):
                path.unlink()
                removed.append(expected_file.path)
        return removed

    def _read_lock(self) -> dict[str, Any]:
        if not self.lock_path.exists():
            return {"version": 1, "models": {}}
        try:
            loaded = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelCacheLockError(f"Corrupt model lock file: {self.lock_path}") from exc
        if not isinstance(loaded, dict):
            raise ModelCacheLockError(f"Invalid model lock file: {self.lock_path}")
        return loaded

    def _write_lock(self, lock: dict[str, Any]) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.lock_path.with_suffix(self.lock_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.lock_path)

    def _lock_entry(self, lock: dict[str, Any], spec: LocalModelSpec) -> dict[str, Any] | None:
        models = lock.get("models")
        if not isinstance(models, dict):
            return None
        entry = models.get(spec.key)
        return entry if isinstance(entry, dict) else None

    def _set_lock_entry(
        self, lock: dict[str, Any], spec: LocalModelSpec, *, status: str, revision: str
    ) -> None:
        lock["version"] = 1
        models = lock.setdefault("models", {})
        models[spec.key] = {
            "hf_id": spec.hf_id,
            "revision": revision,
            "path": spec.path,
            "status": status,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "size_bytes": self._installed_size_bytes(spec),
            "files_checked": status == "ready",
        }

    def _installed_size_bytes(self, spec: LocalModelSpec) -> int:
        model_dir = self._model_dir(spec)
        if not model_dir.exists():
            return 0
        return sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
