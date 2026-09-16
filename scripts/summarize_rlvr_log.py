import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rlvr_diagnostics import parse_metric_records, summarize_rlvr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file")
    args = parser.parse_args()

    path = Path(args.log_file)
    records = parse_metric_records(path.read_text(encoding="utf-8", errors="replace"))
    print(json.dumps(summarize_rlvr(records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
