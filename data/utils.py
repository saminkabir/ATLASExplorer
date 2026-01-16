import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats
from scipy.stats import ttest_rel


def pareto_frontier(x, y, maximize_x=True, maximize_y=True):
    """
    STRICT Pareto frontier in 2D using sort-and-scan, robust to:
      - duplicate x
      - NaN/inf in x or y  (this was causing your crash)
      - empty inputs
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # ✅ Filter NaN/inf pairs FIRST (fixes uniq_x containing nan)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # ✅ If nothing left, return empty (prevents any reduction crash)
    if x.size == 0:
        return np.array([]), np.array([])

    # 1) Collapse duplicates in x: keep best y for each x
    uniq_x = np.unique(x)
    best_y_for_x = np.empty_like(uniq_x, dtype=float)

    # Default value if something weird happens (shouldn’t after finite-mask)
    default = (-np.inf if maximize_y else np.inf)

    for i, xv in enumerate(uniq_x):
        ys = y[x == xv]

        # ✅ extra guard (safe even if nan somehow slips in)
        if ys.size == 0:
            best_y_for_x[i] = default
        else:
            best_y_for_x[i] = ys.max() if maximize_y else ys.min()

    points = np.column_stack([uniq_x, best_y_for_x])

    # Remove any default rows (optional safety)
    keep = np.isfinite(points[:, 1])
    points = points[keep]
    if points.size == 0:
        return np.array([]), np.array([])

    # 2) Sort by x in desired direction
    order = np.argsort(points[:, 0])
    if maximize_x:
        order = order[::-1]
    points = points[order]

    # 3) Strict scan by y
    pareto = []
    if maximize_y:
        best_y = -np.inf
        for xv, yv in points:
            if yv > best_y:  # STRICT improvement
                pareto.append((xv, yv))
                best_y = yv
    else:
        best_y = np.inf
        for xv, yv in points:
            if yv < best_y:  # STRICT improvement
                pareto.append((xv, yv))
                best_y = yv

    pareto = np.array(pareto, dtype=float)
    if pareto.size == 0:
        return np.array([]), np.array([])

    # Return sorted increasing recall (nicer for interpolation/integration)
    s = np.argsort(pareto[:, 0])
    pareto = pareto[s]
    return pareto[:, 0], pareto[:, 1]


def get_recall_data(dataset, model, pareto=True, refinement=1):
    """
    Returns (recall, qps) as numpy arrays (possibly empty).
    Robust to missing dataset/model and NaN/inf rows.
    """
    # NOTE: you print cwd in your current version; keeping it is noisy in Streamlit
    # print(os.getcwd())

    df = pd.read_csv("data/processed/processed_polyvector_results_filtered.csv")
    df = df[(df["dataset"] == dataset) & (df["model"] == model)]

    if df.empty:
        return np.array([]), np.array([])

    # ✅ Drop NaN/inf pairs early (prevents pareto crash)
    df = df[["recall", "qps"]].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return np.array([]), np.array([])

    recall = df["recall"].to_numpy(dtype=float)
    qps = df["qps"].to_numpy(dtype=float)

    if pareto:
        recall, qps = pareto_frontier(recall, qps)

    return np.asarray(recall, dtype=float), np.asarray(qps, dtype=float)


def get_data_for_speedup_recall_graph(dataset, algorithms):
    recalls = []
    qpss = []
    models = []

    refine2_models = set(["scann"])
    refine5_models = set(["VAQ"])

    for algorithm in algorithms:
        refinex = 1
        if algorithm in refine2_models:
            refinex = 2
        if algorithm in refine5_models:
            refinex = 5

        r, q = get_recall_data(dataset, algorithm, refinement=refinex)

        # ✅ Missing pair => just skip points for the curve plot
        if r.size == 0:
            continue

        recalls.extend(r.tolist())
        qpss.extend(q.tolist())
        models.extend([algorithm] * len(r))

    return pd.DataFrame({"recall": recalls, "qps": qpss, "model": models})


def auc_qps_recall(recall, qps, r_min=0.0, r_max=1.0):
    """
    AUC of QPS over recall in [r_min, r_max].
    Returns 0.0 if not enough points.
    """
    recall = np.asarray(recall, dtype=float)
    qps = np.asarray(qps, dtype=float)

    if recall.size < 2 or qps.size < 2:
        return 0.0

    # sort
    order = np.argsort(recall)
    recall = recall[order]
    qps = qps[order]

    # clip range to available domain
    r_min = max(r_min, float(recall[0]))
    r_max = min(r_max, float(recall[-1]))
    if not (r_min < r_max):
        return 0.0

    # create clipped recall grid including boundaries
    r_new = [r_min]
    q_new = [float(np.interp(r_min, recall, qps))]

    for r, q in zip(recall, qps):
        if r_min < r < r_max:
            r_new.append(float(r))
            q_new.append(float(q))

    r_new.append(r_max)
    q_new.append(float(np.interp(r_max, recall, qps)))

    r_new = np.asarray(r_new, dtype=float)
    q_new = np.asarray(q_new, dtype=float)

    return float(np.trapz(q_new, r_new))


def get_data_for_critical_diagram(datasets, algorithms):
    """
    Robust critical-diagram/WCSR data:

    - Missing (dataset, model) pairs => treated as 0.0 (lowest performer)
    - Overlap interval computed among AVAILABLE models only
    - If no overlap among available models => all 0.0 for that dataset

    Returns:
      critical_auc: dict(model -> list of values aligned with critical_datasets)
      critical_datasets: list of datasets actually included
    """

    refine2_models = {"scann"}
    refine5_models = {"VAQ"}

    def _refinement_for(algorithm: str) -> int:
        if algorithm in refine2_models:
            return 2
        if algorithm in refine5_models:
            return 5
        return 1

    # dataset->algorithm->(r,q)
    mp = {}
    critical_auc = {a: [] for a in algorithms}
    critical_datasets = []

    for dataset in datasets:
        # Load curves, track which ones exist
        available = []
        bounds = {}  # algorithm -> (minr, maxr)

        for algorithm in algorithms:
            refinex = _refinement_for(algorithm)
            r, q = get_recall_data(dataset, algorithm, refinement=refinex)
            mp[(dataset, algorithm)] = (r, q)

            if r.size >= 2:
                available.append(algorithm)
                bounds[algorithm] = (float(np.min(r)), float(np.max(r)))

        # If nothing available, skip dataset entirely (or include all zeros; your call)
        if len(available) == 0:
            continue

        # Compute overlap on available algorithms
        minp = max(bounds[a][0] for a in available)
        maxp = min(bounds[a][1] for a in available)

        critical_datasets.append(dataset)

        # No overlap => everyone gets 0.0
        if not (minp < maxp):
            for algorithm in algorithms:
                critical_auc[algorithm].append(0.0)
            continue

        # Overlap exists => compute AUC for available, 0.0 for missing
        for algorithm in algorithms:
            r, q = mp[(dataset, algorithm)]
            if r.size < 2:
                critical_auc[algorithm].append(0.0)
            else:
                critical_auc[algorithm].append(auc_qps_recall(r, q, minp, maxp))

    return critical_auc, critical_datasets
