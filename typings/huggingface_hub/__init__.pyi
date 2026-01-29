"""Type stubs for huggingface_hub."""

from pathlib import Path
from typing import Literal, overload

class HfApi:
    def upload_folder(
        self,
        *,
        folder_path: str | Path,
        path_in_repo: str | None = ...,
        repo_id: str,
        repo_type: str | None = ...,
        allow_patterns: list[str] | str | None = ...,
        ignore_patterns: list[str] | str | None = ...,
        **kwargs: object,
    ) -> str: ...

@overload
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = ...,
    revision: str | None = ...,
    cache_dir: str | Path | None = ...,
    local_dir: str | Path | None = ...,
    library_name: str | None = ...,
    library_version: str | None = ...,
    user_agent: dict[str, str] | str | None = ...,
    force_download: bool = ...,
    token: bool | str | None = ...,
    local_files_only: bool = ...,
    allow_patterns: list[str] | str | None = ...,
    ignore_patterns: list[str] | str | None = ...,
    max_workers: int = ...,
    headers: dict[str, str] | None = ...,
    endpoint: str | None = ...,
    dry_run: Literal[False] = ...,
) -> str: ...

@overload
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = ...,
    revision: str | None = ...,
    cache_dir: str | Path | None = ...,
    local_dir: str | Path | None = ...,
    library_name: str | None = ...,
    library_version: str | None = ...,
    user_agent: dict[str, str] | str | None = ...,
    force_download: bool = ...,
    token: bool | str | None = ...,
    local_files_only: bool = ...,
    allow_patterns: list[str] | str | None = ...,
    ignore_patterns: list[str] | str | None = ...,
    max_workers: int = ...,
    headers: dict[str, str] | None = ...,
    endpoint: str | None = ...,
    dry_run: Literal[True],
) -> list[object]: ...

@overload
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = ...,
    revision: str | None = ...,
    cache_dir: str | Path | None = ...,
    local_dir: str | Path | None = ...,
    library_name: str | None = ...,
    library_version: str | None = ...,
    user_agent: dict[str, str] | str | None = ...,
    force_download: bool = ...,
    token: bool | str | None = ...,
    local_files_only: bool = ...,
    allow_patterns: list[str] | str | None = ...,
    ignore_patterns: list[str] | str | None = ...,
    max_workers: int = ...,
    headers: dict[str, str] | None = ...,
    endpoint: str | None = ...,
    dry_run: bool = ...,
) -> str | list[object]: ...
