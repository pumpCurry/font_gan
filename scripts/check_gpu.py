# -*- coding: utf-8 -*-
"""check_gpu.py — check_gpu

概要:
    PyTorch が GPU を利用可能かを確認する簡単なスクリプト。

:author: pumpCurry
:copyright: (c) pumpCurry 2025 / 5r4ce2
:license: MIT
:version: 1.0.80 (PR #38)
:since:   1.0.58 (PR #26)
:last-modified: 2025-07-13 22:57:24 JST+9
:todo:
    - None
"""

import argparse
import os
import torch


def main() -> None:
    """GPU 利用可否を表示する。"""
    parser = argparse.ArgumentParser(description="Check GPU availability")
    parser.parse_args()
    print(f"{os.path.basename(__file__)} launched.")
    print(torch.__version__, torch.cuda.is_available())


if __name__ == "__main__":
    main()
