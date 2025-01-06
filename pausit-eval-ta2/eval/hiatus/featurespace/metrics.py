"""Implementation of TA1 metrics."""

import copy

import numpy as np
import pandas as pd
from scipy.stats import hmean
from simplification.cutil import simplify_coords
from tqdm.auto import tqdm


def compute_rank_metrics(
    all_needle_ranks,
    n_candidates,
    raw_ranks,
    top_ks=None,
    all_needle_distances=None,
    compute_metrics_at_distance=True,
    plot_resolution=200,
    disable_progress_bars=False,
    **kwargs,
):
    """Compute TA1 metrics."""
    instance = {}
    summary = {}
    plots = {}
    epsilon = 0.0001

    all_needle_ranks_list = all_needle_ranks.to_list()
    all_needle_ranks_tie_averaged = calculate_tied_rank_averages(all_needle_ranks_list, raw_ranks)

    mpr = np.array([np.mean(hits) / n_candidates for hits in all_needle_ranks_tie_averaged])
    instance["Mean Percentile Rank"] = mpr
    summary["Harmonic Mean of Mean Percentile Rank"] = hmean(mpr)

    min_ranks = calculate_nearest_needle_rank(all_needle_ranks_list, raw_ranks)
    instance["Rank of Nearest True Match"] = min_ranks
    summary["Mean Reciprocal Rank"] = 1 / hmean(min_ranks)

    all_n_needles = np.array([len(hits) for hits in all_needle_ranks])
    if top_ks is None:
        top_ks = [1, 8, "all"]

    if "all" in top_ks:
        if n_candidates < 200:
            ranks_ = list(range(1, n_candidates))
        else:
            # Plot first all k less than plot_resolution, then equidistance in log space for another plot_resolution
            ranks_ = np.logspace(2, np.log10(n_candidates), num=plot_resolution, endpoint=False)
            ranks_ = sorted({int(x) for x in ranks_})
            ranks_ = list(range(1, plot_resolution)) + ranks_
        average_success_at_k, average_recall_at_k, average_precision_at_k, average_fpr_at_k = [], [], [], []
        for k in tqdm(ranks_, desc="Computing metrics for many values of k", disable=disable_progress_bars):
            instance_, summary_ = rank_metrics_at_k(all_needle_ranks, all_n_needles, k, n_candidates, raw_ranks)
            average_success_at_k.append(summary_[f"Average Success at {k}"])
            average_recall_at_k.append(summary_[f"Average Recall at {k}"])
            average_precision_at_k.append(summary_[f"Average Precision at {k}"])
            average_fpr_at_k.append(summary_[f"Average FPR at {k}"])
        average_success_at_k.append(1.0)
        average_recall_at_k.append(1.0)
        average_precision_at_k.append(all_n_needles.mean() / n_candidates)
        average_fpr_at_k.append(1.0)
        _ranks = ranks_ + [n_candidates]

        # Simplify plots in log space
        success_coords = [[np.log10(x), y] for x, y in zip(_ranks, average_success_at_k)]
        recall_coords = [[np.log10(x), y] for x, y in zip(_ranks, average_recall_at_k)]
        precision_coords = [[np.log10(x), y] for x, y in zip(_ranks, average_precision_at_k)]
        roc_coords = [[x, y] for x, y in zip(average_fpr_at_k, average_recall_at_k)]
        pvr_coords = [[x, y] for x, y in zip(average_recall_at_k, average_precision_at_k)]

        success_simplified = np.asarray(simplify_coords(success_coords, epsilon))
        recall_simplified = np.asarray(simplify_coords(recall_coords, epsilon))
        precision_simplified = np.asarray(simplify_coords(precision_coords, epsilon))
        roc_simplified = np.asarray(simplify_coords(roc_coords, epsilon))
        pvr_simplified = np.asarray(simplify_coords(pvr_coords, epsilon))

        plots["Average Success at k"] = {
            "x_values": [0] + [int(np.round(10.0**x)) for x in success_simplified[:, 0]],
            "y_values": [0.0] + list(success_simplified[:, 1]),
        }
        plots["Average Recall at k"] = {
            "x_values": [0] + [int(np.round(10.0**x)) for x in recall_simplified[:, 0]],
            "y_values": [0.0] + list(recall_simplified[:, 1]),
        }
        plots["Average Precision at k"] = {
            "x_values": [0] + [int(np.round(10.0**x)) for x in precision_simplified[:, 0]],
            "y_values": [1.0] + list(precision_simplified[:, 1]),
        }
        plots["Average ROC"] = {
            "x_values": [0] + list(roc_simplified[:, 0]),
            "y_values": [0.0] + list(roc_simplified[:, 1]),
        }
        plots["Average Precision vs. Recall"] = {
            "x_values": [0.0] + list(pvr_simplified[:, 0]),
            "y_values": [1.0] + list(pvr_simplified[:, 1]),
        }
        summary.update(
            {
                "Area Under ROC Curve": np.trapz(x=average_fpr_at_k, y=average_recall_at_k),
                "Harmonic Mean of Average Precision": hmean(average_precision_at_k),
            }
        )

    for k in top_ks:
        if k == "all":
            continue
        instance_, summary_ = rank_metrics_at_k(all_needle_ranks, all_n_needles, k, n_candidates, raw_ranks)
        instance.update(instance_)
        summary.update(summary_)

        # Ensure plots contain these points too
        if "all" in top_ks:
            for metric in ["Success", "Recall", "Precision"]:
                if k not in plots[f"Average {metric} at k"]["x_values"]:
                    plots[f"Average {metric} at k"]["x_values"].append(k)
                    plots[f"Average {metric} at k"]["y_values"].append(summary_[f"Average {metric} at {k}"])
                    _x, _y = zip(
                        *sorted(
                            zip(
                                plots[f"Average {metric} at k"]["x_values"],
                                plots[f"Average {metric} at k"]["y_values"],
                            )
                        )
                    )
                    plots[f"Average {metric} at k"]["x_values"] = list(_x)
                    plots[f"Average {metric} at k"]["y_values"] = list(_y)

    if compute_metrics_at_distance and all_needle_distances is not None:
        min_dist = np.array([min(hits) for hits in all_needle_distances])
        instance["Distance to Nearest True Match"] = min_dist
        summary["Mean Distance to Nearest True Match"] = min_dist.mean()

        max_dist = all_needle_distances.apply(max).max()
        delta = max_dist / plot_resolution
        dists_ = np.arange(delta, max_dist, delta)
        average_success_at_k, average_recall_at_k = [0.0], [0.0]
        for dist in tqdm(dists_, desc="Computing metrics for many distances", disable=disable_progress_bars):
            # TODO refactor to pair rank and distance information to get the rest of the retrieval distance metrics
            # rank_metrics_at_k works for for success and recall out-of-the-box
            # but precision and FPR depend on the number of retrieved docs, which is not available here
            instance_, summary_ = rank_metrics_at_k(
                all_needle_distances, all_n_needles, dist, n_candidates, raw_ranks, compute_at_distance=True
            )
            average_success_at_k.append(summary_[f"Average Success at {dist}"])
            average_recall_at_k.append(summary_[f"Average Recall at {dist}"])
        average_success_at_k.append(1.0)
        average_recall_at_k.append(1.0)

        dists_ = [0.0] + list(dists_) + [max_dist]

        success_coords = [[x, y] for x, y in zip(dists_, average_success_at_k)]
        recall_coords = [[x, y] for x, y in zip(dists_, average_recall_at_k)]

        success_simplified = np.asarray(simplify_coords(success_coords, epsilon))
        recall_simplified = np.asarray(simplify_coords(recall_coords, epsilon))

        plots["Average Success at Distance"] = {
            "x_values": list(success_simplified[:, 0]),
            "y_values": list(success_simplified[:, 1]),
        }
        plots["Average Recall at Distance"] = {
            "x_values": list(recall_simplified[:, 0]),
            "y_values": list(recall_simplified[:, 1]),
        }

    return pd.DataFrame(instance), pd.Series(summary, name="Summary"), plots


def rank_metrics_at_k(all_needle_ranks, all_n_needles, k, n_candidates, raw_ranks, compute_at_distance=False):
    """Calculate TA1 metrics at a particular number of retrieved documents."""
    if not compute_at_distance and k > n_candidates:
        raise ValueError(f"Cannot compute metrics at k={k} with only {n_candidates} candidates.")
    instance = {}
    summary = {}
    instance["Total Possible Retrievals"] = all_n_needles

    hits_count = calculate_hits_count(all_needle_ranks, k, raw_ranks)

    instance[f"True Retreivals at {k}"] = hits_count

    # hits at 1 or greater become success of 1
    success = (hits_count >= 1).astype(float)
    # if hits is a fraction between 0 and 1, set success to that fraction
    for i in range(len(success)):
        if success[i] == 0 and hits_count[i] > 0:
            success[i] = hits_count[i]
    instance[f"Success at {k}"] = success
    summary[f"Average Success at {k}"] = success.mean()

    recall = hits_count / all_n_needles
    instance[f"Recall at {k}"] = recall
    summary[f"Average Recall at {k}"] = recall.mean()

    precision = hits_count / k
    instance[f"Precision at {k}"] = precision
    summary[f"Average Precision at {k}"] = precision.mean()

    fpr = (k - hits_count) / (n_candidates - all_n_needles)
    instance[f"FPR at {k}"] = fpr
    summary[f"Average FPR at {k}"] = fpr.mean()

    return instance, summary


def calculate_tied_rank_averages(all_needle_ranks, raw_ranks):
    """Calculate ranks of needles tied with other documents as the average rank of possible ordering permuations."""
    all_needle_ranks_tie_averaged = copy.deepcopy(all_needle_ranks)
    for i in range(len(all_needle_ranks)):  # for every row (query doc)
        tie_aware_row = []
        for needle_rank in all_needle_ranks[i]:  # for every needle, compute average rank
            total_docs_of_rank = len([rank for rank in raw_ranks[i] if rank == needle_rank])
            needle_rank_averaged = (needle_rank + (needle_rank + total_docs_of_rank - 1)) / 2
            tie_aware_row.append(needle_rank_averaged)
        all_needle_ranks_tie_averaged[i] = tie_aware_row
    return all_needle_ranks_tie_averaged


def calculate_hits_count(all_needle_ranks, k, raw_ranks):
    """Calculate tie-aware hits_count for ranks if k is type int, and for distances if k is type np.float."""
    if isinstance(k, np.floating):  # Check whether k is distance (np.float). Otherwise, assume rank (int)
        hits_count = np.array([len([dist for dist in hits if dist <= k]) for hits in all_needle_ranks])
        return hits_count

    # Tied rank calculation: calculate hits at k as the average score of all permutations of the tie
    hits_count = []
    all_needle_ranks = all_needle_ranks.to_list()
    for i in range(len(raw_ranks)):  # iterate over every query document
        ordered_ranks = sorted(raw_ranks[i])
        rank_at_k = ordered_ranks[k - 1]
        needle_docs_at_rank_k = len([rank for rank in all_needle_ranks[i] if rank == rank_at_k])
        if needle_docs_at_rank_k > 0:  # relevant documents on edge of cutoff
            last_ranks_total = len([rank for rank in ordered_ranks if rank == rank_at_k])
            last_ranks_within_k = len([rank for rank in ordered_ranks[:k] if rank == rank_at_k])
            relevance_before_k = len([rank for rank in all_needle_ranks[i] if rank < rank_at_k])
            relevance_at_k = (last_ranks_within_k) * (needle_docs_at_rank_k / last_ranks_total)
            hits_count.append(relevance_before_k + relevance_at_k)
        else:  # no relevant documents on edge of cutoff
            hits_count.append(len([rank for rank in all_needle_ranks[i] if rank <= rank_at_k]))

    hits_count = np.array(hits_count)

    return hits_count


def calculate_nearest_needle_rank(all_needle_ranks, raw_ranks):
    """Calculate tie-averaged rank for 'nearest needle document' reciprocal rank metric."""
    nearest_rank = []
    for i in range(len(all_needle_ranks)):  # iterate over every query document
        rank_of_nearest_hit = min(all_needle_ranks[i])
        total_nearest_ranks = len([rank for rank in raw_ranks[i] if rank == rank_of_nearest_hit])
        if total_nearest_ranks > 1:  # tie is present
            relevant_nearest_ranks = len([rank for rank in all_needle_ranks[i] if rank == rank_of_nearest_hit])
            tied_rank = rank_of_nearest_hit
            frac_of_irrelev_orderings = [1]  # fraction of orderings where all positions up to [i] are haystack docs

            # formula for first position is different than the rest
            frac_of_irrelev_orderings.append(1 - (relevant_nearest_ranks / total_nearest_ranks))

            for j in range(2, total_nearest_ranks + 1):  # remaining positions
                frac_of_irrelev_orderings.append(
                    (1 - relevant_nearest_ranks / (total_nearest_ranks - j + 1)) * frac_of_irrelev_orderings[j - 1]
                )

            # determine fraction of orderings where position [i] (starting at 1) has the first relevant rank
            frac_of_relev_orderings = [0 for i in range(len(frac_of_irrelev_orderings))]
            for i in range(1, len(frac_of_irrelev_orderings)):
                frac_of_relev_orderings[i] = frac_of_irrelev_orderings[i - 1] - frac_of_irrelev_orderings[i]

            # use frac_of_relev_orderings as weight for each rank position when calculating average rank
            instance_nearest_rank = sum(
                [(tied_rank + i - 1) * frac_of_relev_orderings[i] for i in range(1, len(frac_of_relev_orderings))]
            )

            nearest_rank.append(instance_nearest_rank)

        else:
            nearest_rank.append(min(all_needle_ranks[i]))

    return np.array(nearest_rank)
