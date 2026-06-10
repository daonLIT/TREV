"""D3: 서브셋 KS 고유 도메인 빈도목록 생성(tier 화이트리스트 작성 입력).

서브셋(Numerical+Event/Property) 각 claim의 KS URL에서 도메인을 추출해 빈도순으로
집계하고, 현재 config heuristics로 분류한 base tier와 함께 docs/domain_frequency.md에
기록한다. 사람이 이 목록을 보고 config.yaml `tier.overrides`를 채운다.

실행: python -m scripts.domain_freq
"""

from __future__ import annotations

from pathlib import Path

from trev.config import load_config
from trev.data.dataset import load_averitec
from trev.pipeline.tier import classify_domain, subset_domain_frequency

OUT = Path(__file__).resolve().parent.parent / "docs" / "domain_frequency.md"


def main() -> None:
    cfg = load_config()
    include_quote = bool(cfg.get("data", {}).get("include_quote_verification", False))
    tier_cfg = cfg.get("tier", {})

    claims = load_averitec(include_quote=include_quote)
    freq = subset_domain_frequency(claims)

    lines = [
        "# 서브셋 KS 도메인 빈도목록 (D3)",
        "",
        f"- 서브셋 claim 수: {len(claims)}",
        f"- 고유 도메인 수: {len(freq)}",
        "- 빈도 = 서브셋 내 (claim, 고유 URL) 등장 횟수. tier는 현재 config heuristics 분류.",
        "",
        "| # | domain | freq | tier |",
        "| --- | --- | --- | --- |",
    ]
    for i, (domain, count) in enumerate(freq, 1):
        lines.append(f"| {i} | {domain} | {count} | T{classify_domain(domain, tier_cfg)} |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[written] {OUT}  (도메인 {len(freq)}종)")
    print("top 15:")
    for domain, count in freq[:15]:
        print(f"  {count:5d}  T{classify_domain(domain, tier_cfg)}  {domain}")


if __name__ == "__main__":
    main()
