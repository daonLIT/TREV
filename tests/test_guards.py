"""D6 데이터 위생 가드 단위 테스트.

외부 행동만 검증: 입력(텍스트·split·파일명·경로) → 관찰 가능한 출력(예외 발생 여부,
누락 항목 집합). 실 파일시스템·LLM 불필요(레포 `.gitignore`만 실파일로 확인).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trev.guards import (
    BLOCKED_SPLITS,
    REQUIRED_GITIGNORE_ENTRIES,
    DataHygieneError,
    assert_data_file_allowed,
    assert_gitignore_complete,
    assert_knowledge_store_path,
    assert_split_allowed,
    missing_gitignore_entries,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- .gitignore 검증 -------------------------------------------------------

def test_repo_gitignore_has_all_required_entries():
    """실제 레포 .gitignore에 필수 항목이 모두 존재한다(AC: gitignore 포함 확인)."""
    assert_gitignore_complete(REPO_ROOT / ".gitignore")


def test_missing_entries_detected():
    text = "data_store/\n.env\n"  # knowledge_store/·index/·outputs/ 누락
    assert missing_gitignore_entries(text) == ["knowledge_store/", "index/", "outputs/"]


def test_comments_do_not_count_as_entries():
    text = "# index/\n# outputs/\n" + "\n".join(REQUIRED_GITIGNORE_ENTRIES[:2])
    missing = missing_gitignore_entries(text)
    assert "index/" in missing and "outputs/" in missing


def test_complete_text_has_no_missing():
    text = "\n".join(REQUIRED_GITIGNORE_ENTRIES)
    assert missing_gitignore_entries(text) == []


def test_assert_gitignore_complete_raises_on_missing(tmp_path):
    gi = tmp_path / ".gitignore"
    gi.write_text("data_store/\n", encoding="utf-8")
    with pytest.raises(DataHygieneError):
        assert_gitignore_complete(gi)


def test_assert_gitignore_complete_raises_on_absent_file(tmp_path):
    with pytest.raises(DataHygieneError):
        assert_gitignore_complete(tmp_path / "nope.gitignore")


# --- split 가드 ------------------------------------------------------------

def test_dev_split_allowed():
    assert_split_allowed("dev")  # 예외 없음


@pytest.mark.parametrize("split", sorted(BLOCKED_SPLITS))
def test_blocked_splits_raise(split):
    with pytest.raises(DataHygieneError):
        assert_split_allowed(split)


def test_unknown_split_raises():
    with pytest.raises(DataHygieneError):
        assert_split_allowed("validation")


# --- data 파일 가드 --------------------------------------------------------

def test_dev_json_always_allowed():
    assert_data_file_allowed("dev.json")
    assert_data_file_allowed("dev.json", fewshot=True)


def test_train_json_allowed_only_for_fewshot():
    assert_data_file_allowed("train.json", fewshot=True)  # 예외 없음
    with pytest.raises(DataHygieneError):
        assert_data_file_allowed("train.json")  # 평가 경로 차단


@pytest.mark.parametrize("name", ["test.json", "test_2025.json"])
def test_test_files_always_blocked(name):
    with pytest.raises(DataHygieneError):
        assert_data_file_allowed(name, fewshot=True)


# --- knowledge_store 경로 가드 --------------------------------------------

def test_dev_ks_path_allowed():
    assert_knowledge_store_path("knowledge_store/dev/0.json")


@pytest.mark.parametrize("split", sorted(BLOCKED_SPLITS))
def test_blocked_ks_paths_raise(split):
    with pytest.raises(DataHygieneError):
        assert_knowledge_store_path(f"knowledge_store/{split}/0.json")


def test_non_ks_path_raises():
    with pytest.raises(DataHygieneError):
        assert_knowledge_store_path("data_store/averitec/dev.json")
