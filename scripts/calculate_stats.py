# -*- coding: utf-8 -*-
"""calculate_stats.py — scripts.calculate_stats

概要:
    Preprocessed dataset statistics calculator for edge areas.

:author: pumpCurry
:copyright: (c) pumpCurry 2025 / 5r4ce2
:license: MIT
:version: 1.0.72 (PR #33)
:since:   1.0.72 (PR #33)
:last-modified: 2025-06-25 02:48:45 JST+9
:todo:
    - Support additional metrics
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
from tqdm import tqdm
import numpy as np


def main() -> None:
    """Calculate mean edge area from preprocessed dataset."""
    parser = argparse.ArgumentParser(description="Calculate dataset stats")
    parser.add_argument("--preprocessed_dir", type=str, required=True)
    args = parser.parse_args()

    areas: List[float] = []
    for fname in tqdm(os.listdir(args.preprocessed_dir), desc="Calculating"):
        if not fname.endswith(".pt"):
            continue
        data = torch.load(os.path.join(args.preprocessed_dir, fname), map_location="cpu", mmap=True)
        if "edge_area" in data:
            areas.append(float(data["edge_area"]))
    if not areas:
        print("No edge area data found.")
        return

    mean_edge_area = float(np.mean(areas))
    print(f"\n--- Statistics Calculated ---")
    print(f"Mean Edge Area: {mean_edge_area:.6f}")
    print("\n[Action Required]")
    print("Add the following line to your config:")
    print(f"  'mean_edge_area': {mean_edge_area:.6f}")


if __name__ == "__main__":
    main()
