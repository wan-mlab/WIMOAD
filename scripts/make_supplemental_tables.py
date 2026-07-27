#!/usr/bin/env python3
"""Generate WIMOAD supplemental McNemar tables from existing result CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wimoad.tables import (
    build_base_vs_integration_table,
    build_integration_input_vs_fusion_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact McNemar supplemental tables for WIMOAD."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing results_* CSV files.",
    )
    parser.add_argument(
        "--best-models",
        type=Path,
        required=True,
        help="Path to Best_Models_Info/best_models_summary.csv.",
    )
    parser.add_argument(
        "--alpha",
        type=Path,
        required=True,
        help="Path to summary_fused_auc_matched_by_group.csv. Only alpha is used.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for generated tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_table = build_base_vs_integration_table(
        args.results_dir, args.best_models, args.alpha
    )
    integration_input_table = build_integration_input_vs_fusion_table(
        args.results_dir, args.best_models, args.alpha
    )

    base_path = args.out_dir / "supplemental_table_s2_base_vs_integration.csv"
    integration_input_path = (
        args.out_dir / "supplemental_table_s2_integration_input_vs_fusion.csv"
    )
    base_table.to_csv(base_path, index=False)
    integration_input_table.to_csv(integration_input_path, index=False)

    print(f"Wrote {base_path}")
    print(f"Wrote {integration_input_path}")


if __name__ == "__main__":
    main()
