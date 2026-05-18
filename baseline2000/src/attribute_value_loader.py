"""
Loads the merged Llama-3.1-70B vanilla attribute annotations from testRdige
and formats them as human-readable lines for the attribute_aware_with_values prompt.

Joins on row index (the 'index' column in the merged CSV).
"""

import glob
import os

import pandas as pd

import config

SCALE_THRESHOLDS = {attr: (scale / 3, 2 * scale / 3)
                    for attr, scale in config.ATTRIBUTE_SCALES.items()}


def _bin_label(attr, value):
    lo, hi = SCALE_THRESHOLDS[attr]
    if value <= lo:
        return "low"
    if value <= hi:
        return "moderate"
    return "high"


class AttributeValueLoader:
    def __init__(self):
        self._index_map = {}   # row_index → {attr: bin_label, ...}

    def load(self):
        files = glob.glob(config.ATTRIBUTE_VALUES_PATTERN)
        if not files:
            raise FileNotFoundError(
                f"No merged annotations found at: {config.ATTRIBUTE_VALUES_PATTERN}"
            )
        path = max(files, key=os.path.getmtime)
        print(f"[attr_loader]  Loading attribute values: {os.path.basename(path)}")
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")

        for _, row in df.iterrows():
            idx = int(row["index"])
            bins = {}
            for attr in config.ATTRIBUTE_HUMAN_LABELS:
                val = row.get(attr)
                if pd.notna(val):
                    bins[attr] = _bin_label(attr, float(val))
            self._index_map[idx] = bins

        print(f"[attr_loader]  Loaded attribute bins for {len(self._index_map)} rows.")
        return self

    def attribute_lines(self, row_index, annotator_id=None):
        bins = self._index_map.get(int(row_index))
        if not bins:
            return None
        lines = []
        for attr, label in config.ATTRIBUTE_HUMAN_LABELS.items():
            level = bins.get(attr, "unknown")
            lines.append(f"- {label}: {level}")
        return "\n".join(lines)
