#!/usr/bin/env python3
"""
check_k.py - Inspects result JSON files to check if stored 'k' values are integer (floor) or continuous (phi*H).
"""

import sys
import os
import json

def extract_k_values(obj):
    k_vals = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == 'k':
                if isinstance(val, (int, float)):
                    k_vals.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, (int, float)):
                            k_vals.append(item)
            else:
                k_vals.extend(extract_k_values(val))
    elif isinstance(obj, list):
        for item in obj:
            k_vals.extend(extract_k_values(item))
    return k_vals

def resolve_path(filepath):
    if os.path.exists(filepath):
        return filepath
    candidates = [
        os.path.join("rr_results", filepath),
        os.path.join("data", "results", filepath),
        os.path.join("..", "rr_results", filepath),
        os.path.join("..", "data", "results", filepath),
        os.path.join("REVISION_OSN_12OCT2026", "deposit", "rr_results", filepath),
        os.path.join("REVISION_OSN_12OCT2026", "deposit", "data", "results", filepath),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return filepath

def check_file(filepath):
    resolved = resolve_path(filepath)
    filename = os.path.basename(filepath)
    if not os.path.exists(resolved):
        print(f"{filename}: File not found")
        return

    try:
        with open(resolved, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"{filename}: Error loading JSON: {e}")
        return

    k_vals = extract_k_values(data)
    
    if not k_vals:
        print(f"{filename}: no stored k")
        return

    non_integer = [k for k in k_vals if isinstance(k, float) and not k.is_integer()]
    if non_integer:
        print(f"{filename}: k=phi*H (non-integer k present) [{len(non_integer)}/{len(k_vals)} non-integer]")
    else:
        print(f"{filename}: k=floor(phi*H) (all integer) [{len(k_vals)} integer values]")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 check_k.py <file1.json> [file2.json ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        check_file(filepath)

if __name__ == '__main__':
    main()
