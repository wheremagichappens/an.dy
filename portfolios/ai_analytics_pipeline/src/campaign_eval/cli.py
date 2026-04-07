from __future__ import annotations

import argparse
from pathlib import Path

from .config import CampaignConfig
from .data_generation import save_mock_data
from .pipeline import CampaignEvaluationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Campaign evaluation pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-mock-data")
    gen.add_argument("--output-dir", type=Path, default=Path("data/mock"))
    gen.add_argument("--n-users", type=int, default=1500)
    gen.add_argument("--random-state", type=int, default=42)

    run = subparsers.add_parser("run")
    run.add_argument("--data-dir", type=Path, default=Path("data/mock"))
    run.add_argument("--output-dir", type=Path, default=Path("artifacts"))

    args = parser.parse_args()

    if args.command == "generate-mock-data":
        paths = save_mock_data(output_dir=args.output_dir, n_users=args.n_users, random_state=args.random_state)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    config = CampaignConfig(output_dir=args.output_dir)
    pipeline = CampaignEvaluationPipeline(config=config)
    results = pipeline.run_from_csv(args.data_dir)
    pipeline.save_artifacts(results, args.output_dir)
    print(results["summary"]["business_readout"])


if __name__ == "__main__":
    main()
