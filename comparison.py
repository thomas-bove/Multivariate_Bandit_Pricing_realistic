#!/usr/bin/env python3
"""
Full baseline comparison: same CLI as ``experiment_realistic.main`` — all algorithms,
``run_one``, shared CSV, regret PDFs, heatmaps.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pricing_realistic.main_entry import main

if __name__ == "__main__":
    main()
