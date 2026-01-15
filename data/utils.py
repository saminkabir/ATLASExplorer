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
    df=df[(df["dataset"] == dataset) & (df["model"] == model) & (df["refinex"] == refinement)]
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

def get_data_for_critical_diagram(datasets,algorithms):
    refine2_models=set(['scann'])
    refine5_models=set(['VAQ'])
    mp=dict()
    critical_auc={}
    critical_datasets=[]
    for dataset in datasets:
        minp=0
        maxp=1
        for algorithm in algorithms:
            refinex=1
            if algorithm in refine2_models:
                refinex=2
            if algorithm in refine5_models:
                refinex=5
            r,q=get_recall_data(dataset,algorithm,refinement=refinex)
            minp=max(min(r),minp)
            maxp=min(max(r),maxp)
            mp[(dataset,algorithm)]=(r,q)
        # print(datasets,minp,maxp)
        if minp<maxp:
            critical_datasets.append(dataset)
            for algorithm in algorithms:
                v=auc_qps_recall(mp[((dataset,algorithm))][0],mp[((dataset,algorithm))][1],minp,maxp)
                print(dataset,algorithm,minp,maxp,v)
                if algorithm not in critical_auc:
                    critical_auc[algorithm]=[]
                critical_auc[algorithm].append(v)
    return critical_auc,critical_datasets
                
                
                
                
            

# df = pd.read_csv("processed/processed_polyvector_results.csv")
# datasets=set(df["dataset"])
# models=set(df["model"])
# print(datasets)
# print(models)
# xx=[]
# for model in models:
#     xx.append((model,1))
# print(xx)