"""D1 게이트: 실데이터(dev.json) 서브셋 크기·Conflicting 개수·평가 분기 출력.

실행: python -m scripts.subset_gate
"""

from __future__ import annotations

from trev.config import load_config
from trev.dataset import load_averitec, subset_gate


def main() -> None:
    cfg = load_config().get("data", {})
    include_quote = bool(cfg.get("include_quote_verification", False))

    for quote in sorted({False, include_quote}):
        claims = load_averitec(include_quote=quote)
        gate = subset_gate(claims)
        tag = "Numerical+Event/Property" + ("+Quote" if quote else "")
        print(
            f"[{tag}] subset={gate['subset_size']} "
            f"conflicting={gate['conflicting']} "
            f"(threshold {20}) -> eval_mode={gate['eval_mode']}"
        )


if __name__ == "__main__":
    main()
