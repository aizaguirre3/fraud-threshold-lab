"""False-positive characterization.

Counting false positives tells a modelling team nothing they can act on.
Naming the *behaviours* that get mistaken for fraud does. This module isolates
the false alarms at the chosen operating threshold and answers two questions:

1. **Univariate** -- which legitimate populations get flagged at a rate well
   above the overall false-alarm rate? Reported as a lift, so the finding reads
   "legitimate transactions with this attribute are flagged at N times base
   rate", which is directly actionable.
2. **Multivariate** -- what recurring joint profiles do the false alarms fall
   into? A k-means pass over interpretable, standardised features, with each
   cluster profiled against the legitimate population it came from.

Both write structured artifacts. The prose in the README is generated from those
artifacts rather than written by hand, so it cannot drift from the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import AMOUNT, RANDOM_SEED, TARGET

#: Interpretable columns used to describe false alarms. Deliberately excludes
#: the anonymised V/C/D blocks: "V257 is high" is not a finding anyone can act on.
SEGMENT_CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
    "identity_present",
]

SEGMENT_NUMERIC = [
    AMOUNT,
    "amt_log",
    "hour_of_day",
    "grp_card1__count",
    "grp_card1__amt_ratio",
    "grp_addr1__count",
]


def _amount_bucket(amt: pd.Series) -> pd.Series:
    edges = [0, 25, 50, 100, 250, 500, 1000, np.inf]
    labels = ["<$25", "$25-50", "$50-100", "$100-250", "$250-500", "$500-1k", ">$1k"]
    return pd.cut(amt, bins=edges, labels=labels, right=False)


def _hour_bucket(hour: pd.Series) -> pd.Series:
    edges = [0, 6, 12, 18, 24]
    labels = ["00-06", "06-12", "12-18", "18-24"]
    return pd.cut(hour, bins=edges, labels=labels, right=False)


def _card_history_bucket(count: pd.Series) -> pd.Series:
    """How much training history the card had. NaN means the card is new."""
    out = pd.Series("unseen_in_training", index=count.index, dtype=object)
    out[count.notna() & (count <= 5)] = "1-5_prior_txns"
    out[count.notna() & (count > 5) & (count <= 50)] = "6-50_prior_txns"
    out[count.notna() & (count > 50)] = ">50_prior_txns"
    return out


def build_segment_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Interpretable descriptors for segmentation."""
    seg = pd.DataFrame(index=features.index)
    for c in SEGMENT_CATEGORICAL:
        if c in features.columns:
            col = features[c].astype(object)
            seg[c] = col.where(col.notna(), "__missing__").astype(str)
    if AMOUNT in features.columns:
        seg["amount_bucket"] = _amount_bucket(features[AMOUNT]).astype(str)
    if "hour_of_day" in features.columns:
        seg["hour_bucket"] = _hour_bucket(features["hour_of_day"]).astype(str)
    if "grp_card1__count" in features.columns:
        seg["card_history"] = _card_history_bucket(features["grp_card1__count"])
    return seg


def univariate_fp_lift(
    features: pd.DataFrame,
    y_true: np.ndarray,
    flagged: np.ndarray,
    min_segment_size: int = 200,
) -> pd.DataFrame:
    """False-alarm rate by segment, relative to the overall false-alarm rate.

    The denominator is *legitimate transactions only*. That is the honest
    denominator: the question is what share of genuine customers in a segment
    get bothered, not what share of all traffic.
    """
    y = np.asarray(y_true).astype(int)
    flag = np.asarray(flagged).astype(bool)
    legit = y == 0
    seg = build_segment_frame(features)

    base_rate = float(flag[legit].mean())
    rows = []
    for col in seg.columns:
        values = seg.loc[legit, col]
        f = flag[legit]
        for val, mask in values.groupby(values).groups.items():
            pos = values.index.get_indexer(mask)
            n = len(pos)
            if n < min_segment_size:
                continue
            rate = float(f[pos].mean())
            rows.append(
                {
                    "attribute": col,
                    "value": str(val),
                    "legit_transactions": n,
                    "false_alarms": int(f[pos].sum()),
                    "false_alarm_rate": rate,
                    "base_false_alarm_rate": base_rate,
                    "lift_vs_base": rate / base_rate if base_rate > 0 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("lift_vs_base", ascending=False).reset_index(drop=True)


def cluster_false_positives(
    features: pd.DataFrame,
    y_true: np.ndarray,
    flagged: np.ndarray,
    n_clusters: int = 5,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """k-means over the false alarms, profiled against the legitimate population.

    Returns the per-cluster profile table and the cluster label per false
    positive (indexed to the original frame).
    """
    y = np.asarray(y_true).astype(int)
    flag = np.asarray(flagged).astype(bool)
    fp_mask = flag & (y == 0)
    fp_idx = features.index[fp_mask]
    if len(fp_idx) < n_clusters:
        return pd.DataFrame(), pd.Series(dtype=int)

    num_cols = [c for c in SEGMENT_NUMERIC if c in features.columns]
    Xn = features.loc[fp_idx, num_cols]
    Xi = SimpleImputer(strategy="median").fit_transform(Xn)
    Xs = StandardScaler().fit_transform(Xi)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = pd.Series(km.fit_predict(Xs), index=fp_idx, name="fp_cluster")

    legit = features.loc[y == 0]
    seg_all = build_segment_frame(features)

    rows = []
    for c in sorted(labels.unique()):
        idx = labels.index[labels == c]
        sub = features.loc[idx]
        row = {
            "cluster": int(c),
            "n_false_positives": int(len(idx)),
            "share_of_false_positives": len(idx) / len(fp_idx),
        }
        for col in num_cols:
            row[f"{col}__median"] = float(sub[col].median())
            row[f"{col}__legit_median"] = float(legit[col].median())
        # Most over-represented categorical value in this cluster
        for col in seg_all.columns:
            cluster_mode = seg_all.loc[idx, col].mode()
            if len(cluster_mode):
                val = cluster_mode.iloc[0]
                cluster_share = float((seg_all.loc[idx, col] == val).mean())
                legit_share = float((seg_all.loc[legit.index, col] == val).mean())
                row[f"{col}__dominant"] = val
                row[f"{col}__cluster_share"] = cluster_share
                row[f"{col}__legit_share"] = legit_share
                row[f"{col}__over_representation"] = (
                    cluster_share / legit_share if legit_share > 0 else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows), labels


def write_fp_profiles(
    lift: pd.DataFrame,
    clusters: pd.DataFrame,
    operating_point: dict,
    top_n: int = 8,
    min_lift: float = 1.25,
) -> str:
    """Generate the prose profile from the computed tables.

    Written as markdown for direct inclusion in the README. Every number in the
    output is read from the tables, so the prose cannot drift from the run.
    """
    lines: list[str] = []
    lines.append("### False positive profile\n")
    lines.append(
        f"At the operating threshold ({operating_point['threshold']:.4f}), the model "
        f"raises {operating_point['n_alerts']:,} alerts on the test window, of which "
        f"{operating_point['fp']:,} are false alarms "
        f"({100 * operating_point['false_alarm_rate']:.1f}% of alerts). "
        f"Against the {operating_point['n_legit']:,} legitimate transactions in the "
        f"window that is a base false-alarm rate of "
        f"{100 * operating_point['base_false_alarm_rate']:.2f}%.\n"
    )

    if not lift.empty:
        top = lift[lift["lift_vs_base"] >= min_lift].head(top_n)
        if len(top):
            lines.append("**Legitimate populations flagged above base rate**\n")
            for _, r in top.iterrows():
                lines.append(
                    f"- Legitimate transactions where `{r['attribute']}` = "
                    f"`{r['value']}` are flagged at **{r['lift_vs_base']:.2f}x base rate** "
                    f"({100 * r['false_alarm_rate']:.2f}% vs "
                    f"{100 * r['base_false_alarm_rate']:.2f}%), "
                    f"accounting for {int(r['false_alarms']):,} false alarms across "
                    f"{int(r['legit_transactions']):,} transactions."
                )
            lines.append("")
        else:
            lines.append(
                f"No single attribute value reaches {min_lift}x the base false-alarm "
                "rate at a segment size of 200 or more. The false positives are "
                "diffuse rather than concentrated in one behaviour.\n"
            )

    if not clusters.empty:
        lines.append("**Recurring joint profiles**\n")
        for _, r in clusters.iterrows():
            desc = []
            for col in SEGMENT_CATEGORICAL + ["amount_bucket", "hour_bucket", "card_history"]:
                key = f"{col}__dominant"
                over_key = f"{col}__over_representation"
                if key in r and pd.notna(r.get(over_key)) and r[over_key] >= 1.2:
                    desc.append(
                        f"{col}=`{r[key]}` ({r[over_key]:.1f}x over-represented)"
                    )
            amt_med = r.get(f"{AMOUNT}__median")
            amt_legit = r.get(f"{AMOUNT}__legit_median")
            amt_txt = ""
            if amt_med is not None and amt_legit is not None:
                amt_txt = (
                    f" Median amount ${amt_med:,.2f} against a legitimate median of "
                    f"${amt_legit:,.2f}."
                )
            lines.append(
                f"- **Cluster {int(r['cluster'])}** — {int(r['n_false_positives']):,} "
                f"false alarms ({100 * r['share_of_false_positives']:.1f}% of all false "
                f"alarms)."
                + (f" Distinguished by {'; '.join(desc)}." if desc else "")
                + amt_txt
            )
        lines.append("")
    return "\n".join(lines)
