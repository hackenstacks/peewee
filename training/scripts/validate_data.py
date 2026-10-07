#!/usr/bin/env python3
# Author  : hackenstacks@protonmail.com
# Contact : hackenstacks@protonmail.com
"""Quick validation of training data before upload."""

import json
import pathlib
import sys

DATA_FILE = pathlib.Path(__file__).parent.parent / "data" / "nexus_train_mistral.jsonl"

def validate():
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found")
        sys.exit(1)

    total = 0
    bad = 0
    role_counts = {"system": 0, "user": 0, "assistant": 0}

    with open(DATA_FILE) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                if not msgs:
                    bad += 1
                    continue
                has_user = any(m.get("role") == "user" for m in msgs)
                has_asst = any(m.get("role") == "assistant" for m in msgs)
                if not (has_user and has_asst):
                    bad += 1
                    continue
                for m in msgs:
                    role = m.get("role", "unknown")
                    if role in role_counts:
                        role_counts[role] += 1
                total += 1
            except json.JSONDecodeError:
                bad += 1

    print(f"Total valid samples : {total:,}")
    print(f"Bad/skipped         : {bad:,}")
    print(f"Role distribution   : {role_counts}")
    print(f"File size           : {DATA_FILE.stat().st_size / 1024 / 1024:.1f} MB")

    if bad > total * 0.05:
        print("WARNING: >5% bad samples — review before upload")
    else:
        print("OK: Data looks clean for upload")

if __name__ == "__main__":
    validate()
