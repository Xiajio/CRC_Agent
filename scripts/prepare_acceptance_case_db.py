from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.testing.acceptance_case_db import materialize_acceptance_case_db


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the deterministic acceptance case database.")
    parser.add_argument("--seed", required=True, help="Path to the JSON seed file.")
    parser.add_argument("--output", required=True, help="Directory to write the database into.")
    args = parser.parse_args()

    materialize_acceptance_case_db(seed_path=Path(args.seed), output_root=Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

