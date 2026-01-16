import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from autorank import autorank, plot_stats
from scipy.stats import ttest_rel
import pandas as pd

def pareto_frontier(x, y, maximize_x=True, maximize_y=True):
    """
    STRICT Pareto frontier in 2D using sort-and-scan, robust to duplicate x's.

    Strict here means: after sorting by x (best to worst), we only keep a point
    if it STRICTLY improves y over all previously kept points.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # 1) Collapse duplicates in x: keep best y for each x
    #    (otherwise points with same x can cause confusing behavior)
    uniq_x = np.unique(x)
    best_y_for_x = np.empty_like(uniq_x, dtype=float)

    for i, xv in enumerate(uniq_x):
        ys = y[x == xv]
        best_y_for_x[i] = ys.max() if maximize_y else ys.min()

    points = np.column_stack([uniq_x, best_y_for_x])

    # 2) Sort by x in desired direction
    order = np.argsort(points[:, 0])
    if maximize_x:
        order = order[::-1]
    points = points[order]

    # 3) Strict scan by y (compare to best-so-far y)
    pareto = []
    if maximize_y:
        best_y = -np.inf
        for xv, yv in points:
            if yv > best_y:          # STRICT improvement
                pareto.append((xv, yv))
                best_y = yv
    else:
        best_y = np.inf
        for xv, yv in points:
            if yv < best_y:          # STRICT improvement
                pareto.append((xv, yv))
                best_y = yv

    pareto = np.array(pareto, dtype=float)
    return pareto[:, 0], pareto[:, 1]

import os

def get_recall_data(dataset,model,pareto=True,refinement=1):
    print(os.getcwd())
    df = pd.read_csv("data/processed/processed_polyvector_results_filtered.csv")
    df=df[(df["dataset"] == dataset) & (df["model"] == model)]
    recall,qps = df['recall'].tolist(),df['qps'].tolist()
    if len(recall)==0:
        return recall,qps
    if pareto:
        recall,qps=pareto_frontier(recall,qps)
    return recall,qps

def get_data_for_speedup_recall_graph(dataset,algorithms):
    recalls=[]
    qpss=[]
    models=[]
    refine2_models=set(['scann'])
    refine5_models=set(['VAQ'])
    for algorithm in algorithms:
        refinex=1
        if algorithm in refine2_models:
            refinex=2
        if algorithm in refine5_models:
            refinex=5
        r,q=get_recall_data(dataset,algorithm,refinement=refinex)
        for i in range(0,len(r)):
            recalls.append(r[i])
            qpss.append(q[i])
            models.append(algorithm)
    df = pd.DataFrame({
        "recall": recalls,
        "qps": qpss,
        "model": models
    })
    return df

def auc_qps_recall(recall, qps, r_min=0.0, r_max=1.0):
    recall = np.array(recall, dtype=float)
    qps = np.array(qps, dtype=float)

    # sort
    order = np.argsort(recall)
    recall = recall[order]
    qps = qps[order]

    # clip range to available domain
    r_min = max(r_min, recall[0])
    r_max = min(r_max, recall[-1])

    # create clipped recall grid including boundaries
    r_new = [r_min]
    q_new = [np.interp(r_min, recall, qps)]

    for r, q in zip(recall, qps):
        if r_min < r < r_max:
            r_new.append(r)
            q_new.append(q)

    r_new.append(r_max)
    q_new.append(np.interp(r_max, recall, qps))

    r_new = np.array(r_new)
    q_new = np.array(q_new)

    # integrate
    return np.trapz(q_new, r_new)



def get_data_for_critical_diagram(datasets, algorithms):
    """
    Per-dataset behavior:
      - We try to compute the maximally-overlapped recall interval [minp, maxp]
        across ALL selected algorithms on that dataset.
      - If the interval exists (minp < maxp): compute AUC per algorithm on [minp, maxp].
      - If the interval does NOT exist (minp >= maxp): we still keep the dataset and
        assign AUC=0.0 to ONLY the algorithm(s) responsible for breaking the overlap
        (the blockers). All other algorithms get their AUC computed on the overlap
        of the remaining algorithms (i.e., after excluding the blockers).
        If even after removing the blockers there is still no overlap, we fall back
        to assigning 0.0 to all algorithms for that dataset.

    Also returns a nonoverlap_report dict for debugging/UI.

    Requires:
      - get_recall_data(dataset, algorithm, refinement=1) -> (r_list, q_list)
      - auc_qps_recall(r_list, q_list, minp, maxp) -> float
    """

    # --- your refinement mapping ---
    refine2_models = {"scann"}
    refine5_models = {"VAQ"}

    def _refinement_for(algorithm: str) -> int:
        if algorithm in refine2_models:
            return 2
        if algorithm in refine5_models:
            return 5
        return 1

    def _bounds(r_list):
        return (min(r_list), max(r_list))

    def _intersection_interval(bounds_dict, subset_algs):
        """bounds_dict[a] = (min_r, max_r)"""
        minp = max(bounds_dict[a][0] for a in subset_algs)
        maxp = min(bounds_dict[a][1] for a in subset_algs)
        return minp, maxp

    def diagnose_nonoverlap(bounds_dict, eps=1e-12):
        """
        bounds_dict[a] = (min_r, max_r)
        Returns overlap interval and which algorithms are blockers (responsible).
        """
        mins = {a: bounds_dict[a][0] for a in bounds_dict}
        maxs = {a: bounds_dict[a][1] for a in bounds_dict}

        max_of_mins = max(mins.values())
        min_of_maxs = min(maxs.values())

        has_overlap = (max_of_mins + eps) < min_of_maxs

        lower_blockers = [a for a, v in mins.items() if abs(v - max_of_mins) <= eps]
        upper_blockers = [a for a, v in maxs.items() if abs(v - min_of_maxs) <= eps]

        blockers = sorted(set(lower_blockers + upper_blockers))

        return {
            "mins": mins,
            "maxs": maxs,
            "intersection": (max_of_mins, min_of_maxs),
            "has_overlap": has_overlap,
            "lower_blockers": lower_blockers,
            "upper_blockers": upper_blockers,
            "blockers": blockers,
        }

    # --- main computation ---
    mp = {}  # (dataset, algorithm) -> (r, q)
    critical_auc = {}  # algorithm -> list of auc values (aligned with critical_datasets)
    critical_datasets = []
    nonoverlap_report = {}  # dataset -> diagnostics

    for dataset in datasets:
        # 1) load all curves + bounds for this dataset
        bounds = {}
        for algorithm in algorithms:
            refinex = _refinement_for(algorithm)
            r, q = get_recall_data(dataset, algorithm, refinement=refinex)
            mp[(dataset, algorithm)] = (r, q)
            bounds[algorithm] = _bounds(r)

        # 2) compute global overlap across all algorithms
        minp_all, maxp_all = _intersection_interval(bounds, algorithms)

        critical_datasets.append(dataset)

        # Ensure lists exist and we'll append exactly one value per dataset
        for algorithm in algorithms:
            critical_auc.setdefault(algorithm, [])

        # Case A: overlap exists across all -> normal behavior
        if minp_all < maxp_all:
            for algorithm in algorithms:
                r, q = mp[(dataset, algorithm)]
                v = auc_qps_recall(r, q, minp_all, maxp_all)
                critical_auc[algorithm].append(v)
            nonoverlap_report[dataset] = {
                "status": "ok",
                "interval_used": (minp_all, maxp_all),
                "blockers": [],
            }
            continue

        # Case B: no overlap across all -> find blockers and compute on "others" interval
        info = diagnose_nonoverlap(bounds)
        blockers = info["blockers"]

        # Compute overlap among NON-blockers (the "rest")
        remaining = [a for a in algorithms if a not in blockers]

        # If everybody is a blocker (edge case), no remaining
        if len(remaining) == 0:
            for algorithm in algorithms:
                critical_auc[algorithm].append(0.0)
            nonoverlap_report[dataset] = {
                "status": "no_overlap_all_blockers",
                "global_intersection": info["intersection"],
                "lower_blockers": info["lower_blockers"],
                "upper_blockers": info["upper_blockers"],
                "blockers": blockers,
                "interval_used_for_nonblockers": None,
            }
            continue

        minp_rest, maxp_rest = _intersection_interval(bounds, remaining)

        # If remaining still has no overlap, fall back to all zeros
        if not (minp_rest < maxp_rest):
            for algorithm in algorithms:
                critical_auc[algorithm].append(0.0)
            nonoverlap_report[dataset] = {
                "status": "no_overlap_even_without_blockers",
                "global_intersection": info["intersection"],
                "lower_blockers": info["lower_blockers"],
                "upper_blockers": info["upper_blockers"],
                "blockers": blockers,
                "interval_used_for_nonblockers": (minp_rest, maxp_rest),
            }
            continue

        # Otherwise:
        #   - blockers get 0.0 (lowest performers)
        #   - non-blockers get real AUC on the remaining overlap interval
        for algorithm in algorithms:
            if algorithm in blockers:
                critical_auc[algorithm].append(0.0)
            else:
                r, q = mp[(dataset, algorithm)]
                v = auc_qps_recall(r, q, minp_rest, maxp_rest)
                critical_auc[algorithm].append(v)

        nonoverlap_report[dataset] = {
            "status": "no_overlap_fixed_by_zeroing_blockers",
            "global_intersection": info["intersection"],
            "lower_blockers": info["lower_blockers"],
            "upper_blockers": info["upper_blockers"],
            "blockers": blockers,
            "interval_used_for_nonblockers": (minp_rest, maxp_rest),
        }

    return critical_auc, critical_datasets


                
                
                
                
            

# df = pd.read_csv("processed/processed_polyvector_results.csv")
# datasets=set(df["dataset"])
# models=set(df["model"])
# print(datasets)
# print(models)
# xx=[]
# for model in models:
#     xx.append((model,1))
# print(xx)