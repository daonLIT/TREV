"""D5: gold URL ↔ top-k Recall 매칭(URL 정규화) + 검색실패 vs 실제NEI 구분.

검색된 top-k 문서 url을 gold `questions[].answers[].source_url`과 **정규화 후 매칭**한다:
아카이브 prefix 제거(→원본 URL), 스킴/www/쿼리/fragment/끝슬래시 제거, 소문자 host.
gold가 KS에 있는데 미회수면 '검색실패', gold label이 NEI면 '실제부재'로 구분한다.
(시스템 슬라이스 S7/#9가 소비.)
"""

from __future__ import annotations

from urllib.parse import urlsplit

from trev.data.knowledge_store import recover_archive_url
from trev.schemas import AveritecLabel


def normalize_url(url: str | None) -> str:
    """매칭용 정규화: 아카이브 원본 복원 + 스킴/www/쿼리/fragment/끝슬래시 제거."""
    if not url:
        return ""
    original = recover_archive_url(url)
    parts = urlsplit(original if "://" in original else "http://" + original)
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[len("www."):]
    return host + parts.path.rstrip("/")


def _norm_set(urls) -> set[str]:
    return {n for u in urls if (n := normalize_url(u))}


def recall_at_k(retrieved_urls, gold_urls, k: int) -> float | None:
    """top-k url ∩ gold / |gold|. gold 없으면 None(Recall 미정의)."""
    gold = _norm_set(gold_urls)
    if not gold:
        return None
    top = _norm_set(retrieved_urls[:k])
    return len(gold & top) / len(gold)


def precision_at_k(retrieved_urls, gold_urls, k: int) -> float | None:
    """top-k 중 gold 비율. 회수 결과 없으면 None."""
    gold = _norm_set(gold_urls)
    top = [normalize_url(u) for u in retrieved_urls[:k] if normalize_url(u)]
    if not top:
        return None
    return sum(1 for u in top if u in gold) / len(top)


def classify_retrieval(
    gold_urls,
    ks_urls,
    retrieved_urls,
    *,
    gold_label: AveritecLabel | None,
    k: int,
) -> str:
    """검색 결과 원인을 분류한다(오류 원인 분리).

    - retrieved: gold가 top-k에 회수됨.
    - retrieval_failure: gold가 KS에 있는데 top-k 미회수(=검색 실패).
    - gold_absent_from_ks: gold가 KS에도 없음(코퍼스 한계).
    - true_nei: gold 근거 URL 없음 & gold label=NEI(실제 부재).
    - no_gold_urls: gold 근거 URL 없음(라벨이 NEI도 아님).
    """
    gold = _norm_set(gold_urls)
    if gold:
        top = _norm_set(retrieved_urls[:k])
        if gold & top:
            return "retrieved"
        return (
            "retrieval_failure"
            if gold & _norm_set(ks_urls)
            else "gold_absent_from_ks"
        )
    return "true_nei" if gold_label is AveritecLabel.NOT_ENOUGH_EVIDENCE else "no_gold_urls"
