"""
Loads the LLM's own previous attribute predictions for the
`attribute_aware_with_values` prompt.

Source: latest merged_standard_results_*.csv from FullAnnotation_<model>/results/
Joined on (comment_id, annotator_id). Each row's numeric value is converted to
{low, moderate, high} by binning into thirds of the attribute's native scale.
"""

import os
import glob

import pandas as pd

import config


def _bin_label(value, max_value):
    """Return 'low' / 'moderate' / 'high' for a numeric value on a [0, max] scale."""
    if value is None or pd.isna(value):
        return "unknown"
    third = max_value / 3.0
    if value <= third:
        return "low"
    if value <= 2 * third:
        return "moderate"
    return "high"


class AttributeValueLoader:
    def __init__(self, pattern=None):
        self.pattern = pattern or config.ATTRIBUTE_VALUES_PATTERN
        self.df = None

    def load(self):
        files = glob.glob(self.pattern)
        if not files:
            raise FileNotFoundError(
                f"No attribute-prediction CSV matched: {self.pattern}\n"
                f"Run the FullAnnotation pipeline first."
            )
        latest = max(files, key=os.path.getmtime)
        print(f"[attr-values] Loading: {os.path.basename(latest)}")
        self.df = pd.read_csv(latest, low_memory=False)

        cols_needed = ["comment_id", "annotator_id"]
        for c in cols_needed:
            if c not in self.df.columns:
                raise ValueError(f"Required column missing in attribute file: {c}")

        for attr in config.ATTRIBUTE_HUMAN_LABELS:
            if attr in self.df.columns:
                self.df[attr] = pd.to_numeric(self.df[attr], errors="coerce")

        self.df = self.df.set_index(["comment_id", "annotator_id"], drop=False)
        return self

    def attribute_lines(self, comment_id, annotator_id):
        """Return a multi-line string '- Attack/Defend: high\\n- ...' for this row."""
        if self.df is None:
            return None
        try:
            row = self.df.loc[(comment_id, annotator_id)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        except KeyError:
            return None

        lines = []
        for attr, label in config.ATTRIBUTE_HUMAN_LABELS.items():
            if attr not in row.index:
                continue
            max_v = config.ATTRIBUTE_SCALES.get(attr, 4)
            bin_lbl = _bin_label(row[attr], max_v)
            lines.append(f"- {label}: {bin_lbl}")
        return "\n".join(lines)
