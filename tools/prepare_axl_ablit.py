#!/usr/bin/env python3
"""Reproduce, build, and verify the pinned AXL-ABLIT checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm53_setup.axl_ablit import AxlAblitError, build, reproduce_axl, verify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    reproduce = commands.add_parser(
        "reproduce", help="run the mandatory all-29 NVIDIA-to-public-AXL gate"
    )
    reproduce.add_argument("--nvidia", required=True, type=Path)
    reproduce.add_argument("--axl", required=True, type=Path)
    reproduce.add_argument("--output", required=True, type=Path)

    materialize = commands.add_parser(
        "build", help="materialize an atomic derived checkpoint"
    )
    materialize.add_argument("--axl", required=True, type=Path)
    materialize.add_argument("--nvidia", required=True, type=Path)
    materialize.add_argument("--donor-overlay", required=True, type=Path)
    materialize.add_argument(
        "--reproduction-report",
        required=True,
        type=Path,
        help="output path; build always regenerates this report",
    )
    materialize.add_argument("--output", required=True, type=Path)

    check = commands.add_parser("verify", help="verify a complete derived checkpoint")
    check.add_argument("--checkpoint", required=True, type=Path)
    check.add_argument("--base", type=Path)
    check.add_argument("--manifest-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "reproduce":
            result = reproduce_axl(args.nvidia, args.axl, args.output)
            summary = {
                "passed": result["passed"],
                "tensor_count": result["tensor_count"],
                "report": str(args.output),
            }
        elif args.command == "build":
            result = build(
                args.axl,
                args.nvidia,
                args.donor_overlay,
                args.reproduction_report,
                args.output,
            )
            summary = {
                "complete": result["complete"],
                "transformed_key_count": result["transformed_key_count"],
                "output": str(args.output),
            }
        else:
            summary = verify(
                args.checkpoint,
                base=args.base,
                expected_manifest_sha256=args.manifest_sha256,
            )
    except (AxlAblitError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
