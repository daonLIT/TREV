"""데이터 위생 가드 (cross-cutting).

AVeriTeC CC BY-NC 4.0(데이터 재배포 금지)과 실험 무결성(blind split 차단)을
코드 레벨에서 강제한다. D1/D2 등 이후 로더가 이 가드를 호출하는 seam이다.

규칙
- 평가/검색에 쓸 수 있는 split은 dev 뿐.
- test / test_2025 / train KS는 로드 시도 자체를 차단(blind·Out of Scope).
- train.json은 few-shot 예시 경로(fewshot=True)로만 접근 허용, 평가 경로 차단.
- 민감 산출물 디렉터리(data_store/·knowledge_store/·index/·outputs/·.env)는
  .gitignore에 반드시 포함(데이터 재배포 금지).
"""

from __future__ import annotations

from pathlib import Path

# .gitignore에 반드시 존재해야 하는 항목(데이터·산출물·비밀키 재배포 차단).
REQUIRED_GITIGNORE_ENTRIES: tuple[str, ...] = (
    "data_store/",
    "knowledge_store/",
    "index/",
    "outputs/",
    ".env",
)

# 검색·평가에 허용되는 데이터 split.
ALLOWED_SPLITS: frozenset[str] = frozenset({"dev"})

# 로드 자체가 금지된 split(blind·Out of Scope).
BLOCKED_SPLITS: frozenset[str] = frozenset({"test", "test_2025", "train"})


class DataHygieneError(Exception):
    """허용되지 않은 데이터 split·파일·경로 접근 시 발생."""


def missing_gitignore_entries(gitignore_text: str) -> list[str]:
    """`gitignore_text`에서 빠진 필수 항목을 순서대로 반환한다(없으면 빈 리스트).

    주석(`#`)·공백 라인은 무시한다. 루트 앵커(`/data_store/`)와 비앵커(`data_store/`)를 동일하게
    인정하기 위해 앞 슬래시는 무시하고 비교한다.
    """
    present = {
        line.strip().lstrip("/")
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return [entry for entry in REQUIRED_GITIGNORE_ENTRIES if entry.lstrip("/") not in present]


def assert_gitignore_complete(gitignore_path: str | Path) -> None:
    """`.gitignore`에 필수 항목이 모두 있는지 검증한다(없으면 예외)."""
    path = Path(gitignore_path)
    if not path.is_file():
        raise DataHygieneError(f".gitignore를 찾을 수 없음: {path}")
    missing = missing_gitignore_entries(path.read_text(encoding="utf-8"))
    if missing:
        raise DataHygieneError(
            ".gitignore에 누락된 필수 항목(데이터 재배포 위험): " + ", ".join(missing)
        )


def assert_split_allowed(split: str) -> None:
    """검색·평가에 쓰는 split이 허용 범위(dev)인지 검증한다.

    test / test_2025 / train은 blind·Out of Scope이므로 예외를 던진다.
    """
    if split in ALLOWED_SPLITS:
        return
    if split in BLOCKED_SPLITS:
        raise DataHygieneError(
            f"'{split}' split은 blind·Out of Scope라 로드 금지(허용: dev)."
        )
    raise DataHygieneError(f"알 수 없는 split '{split}' (허용: {sorted(ALLOWED_SPLITS)}).")


def assert_data_file_allowed(filename: str, *, fewshot: bool = False) -> None:
    """`data_store/averitec/*.json` 접근을 검증한다.

    - dev.json: 항상 허용.
    - train.json: few-shot 예시 경로(`fewshot=True`)로만 허용, 평가 경로 차단.
    - test.json / test_2025.json: 항상 차단.
    """
    stem = Path(filename).stem  # "dev.json" -> "dev"
    if stem == "dev":
        return
    if stem == "train":
        if fewshot:
            return
        raise DataHygieneError(
            "train.json은 few-shot 예시 전용(fewshot=True)이며 평가 경로 접근 금지."
        )
    if stem in BLOCKED_SPLITS:
        raise DataHygieneError(f"{filename}은 blind·Out of Scope라 접근 금지.")
    raise DataHygieneError(f"알 수 없는 데이터 파일 '{filename}'.")


def assert_knowledge_store_path(path: str | Path) -> None:
    """`knowledge_store/<split>/...` 경로의 split이 허용 범위인지 검증한다."""
    parts = Path(path).parts
    if "knowledge_store" not in parts:
        raise DataHygieneError(f"knowledge_store 경로가 아님: {path}")
    idx = parts.index("knowledge_store")
    if idx + 1 >= len(parts):
        raise DataHygieneError(f"knowledge_store split을 특정할 수 없음: {path}")
    assert_split_allowed(parts[idx + 1])
