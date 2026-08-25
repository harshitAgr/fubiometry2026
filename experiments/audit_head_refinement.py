"""Verify that checkpoint refinement changed only one task head."""
from __future__ import annotations

import argparse
import json
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    before = torch.load(args.before, map_location="cpu", weights_only=True)
    after = torch.load(args.after, map_location="cpu", weights_only=True)
    if before.keys() != after.keys():
        missing = sorted(before.keys() - after.keys())
        added = sorted(after.keys() - before.keys())
        raise ValueError(f"state_dict keys differ; missing={missing}, added={added}")

    changed = [key for key in before if not torch.equal(before[key], after[key])]
    prefix = f"heads.{args.task}."
    unexpected = [key for key in changed if not key.startswith(prefix)]
    result = {
        "before": args.before,
        "after": args.after,
        "task": args.task,
        "n_state_tensors": len(before),
        "n_changed_tensors": len(changed),
        "changed_tensors": changed,
        "unexpected_changed_tensors": unexpected,
        "only_requested_head_changed": bool(changed) and not unexpected,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
    if not result["only_requested_head_changed"]:
        raise SystemExit("checkpoint changed outside the requested task head, or changed nothing")


if __name__ == "__main__":
    main()
