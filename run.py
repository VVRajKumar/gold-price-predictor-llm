"""
CLI runner – generate a prediction from the command line.
Usage:  python run.py [--once | --daemon]
"""

import argparse
import sys
import json
from pathlib import Path

# Fix imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Streamlit Cloud: repos live at /mount/src/<name>/ which can shadow our 'src' pkg
_expected = ROOT / "src" / "__init__.py"
if "src" in sys.modules:
    _loaded = getattr(sys.modules["src"], "__file__", None)
    if _loaded is None or Path(_loaded).resolve() != _expected.resolve():
        for _k in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
            del sys.modules[_k]

from src.prediction_engine import PredictionEngine
from loguru import logger

logger.add("logs/gold_predictor.log", rotation="10 MB", retention="30 days")


def main():
    parser = argparse.ArgumentParser(description="Gold Price Predictor – Agentic AI")
    parser.add_argument(
        "--once", action="store_true",
        help="Generate a single prediction and exit",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run in daemon mode with auto-refresh",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output prediction as JSON",
    )
    args = parser.parse_args()

    engine = PredictionEngine()

    if args.daemon:
        logger.info("Starting in daemon mode …")
        engine.start_auto_refresh()
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            engine.stop_auto_refresh()
            logger.info("Stopped.")
    else:
        plan = engine.generate()

        if args.json:
            print(plan.model_dump_json(indent=2))
        else:
            print("\n" + "=" * 70)
            print("  🥇 GOLD PRICE PREDICTION – AGENTIC AI SYSTEM")
            print("=" * 70)
            print(f"\n  Current Price:  ${plan.current_price:,.2f}")
            print(f"  Outlook:        {plan.overall_outlook.upper()}")
            print(f"  Confidence:     {plan.overall_confidence:.0%}")
            print(f"  Generated:      {plan.generated_at}")
            print(f"\n  {'─' * 60}")
            print(f"\n  EXECUTIVE SUMMARY:")
            print(f"  {plan.executive_summary[:500]}")
            print(f"\n  {'─' * 60}")
            print(f"\n  7-DAY PREDICTIONS:")
            print(f"  {'Date':<12} {'Predicted':>10} {'Low':>10} {'High':>10} {'Conf':>6}  Driver")
            print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*6}  {'─'*20}")
            for dp in plan.daily_predictions:
                print(
                    f"  {dp.date:<12} ${dp.predicted_price:>9,.2f} "
                    f"${dp.low_range:>9,.2f} ${dp.high_range:>9,.2f} "
                    f"{dp.confidence:>5.0%}  {dp.key_driver[:30]}"
                )

            print(f"\n  {'─' * 60}")
            print(f"\n  AGENT CONSENSUS:")
            for name, report in plan.agent_reports.items():
                if isinstance(report, dict) and "outlook" in report:
                    bias = report.get("prediction_bias", 0)
                    arrow = "▲" if bias > 0 else "▼" if bias < 0 else "►"
                    print(
                        f"  {arrow} {name:<30s} {report['outlook']:<10s} "
                        f"conf={report.get('confidence', 0):.0%} bias={bias:+.2f}"
                    )

            if plan.risk_factors:
                print(f"\n  {'─' * 60}")
                print(f"\n  RISK FACTORS:")
                for i, r in enumerate(plan.risk_factors, 1):
                    print(f"  {i}. {r}")

            print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
