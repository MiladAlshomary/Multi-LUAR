"""Implementation of TA2 metrics."""

import warnings

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from simplification.cutil import simplify_coords
from sklearn.metrics import det_curve, precision_recall_fscore_support, roc_auc_score, roc_curve


warnings.filterwarnings("once")


def compute_attribution_metrics(ground_truth, scores, predictions, far_target=None, which_metrics="all"):
    """Compute TA2 metrics."""
    instance = []
    plots = {}
    for idx in ground_truth.index:
        instance_, plots_ = instance_attribution_metrics(
            scores.loc[[idx]].to_numpy()[0],
            ground_truth.loc[[idx]].to_numpy()[0],
            predictions.loc[[idx]].to_numpy()[0],
            far_target,
            which_metrics,
        )
        instance.append(instance_)
        plots[idx] = plots_

    instance = pd.concat(instance, axis=1).T
    instance.index = ground_truth.index

    summary, sum_plts = instance_attribution_metrics(
        scores.to_numpy().flatten(),
        ground_truth.to_numpy().flatten(),
        predictions.to_numpy().flatten(),
        far_target,
        which_metrics,
    )
    summary.name = "Summary"
    if "all" in which_metrics or "tar_at_far" in which_metrics or "det" in which_metrics:
        plots["Summary"] = sum_plts
        plots = format_to_plot(plots, eer_to_transform=summary["Equal Error Rate"])
    return instance, summary, plots


def format_to_plot(diction, metric_names=["DET", "Tar@Far"], eer_to_transform=None):
    """Convert plot-able outputs to proper format."""
    new_plot = {met_name: {} for met_name in metric_names}
    for author, values in diction.items():
        for metric, vals in values.items():
            try:
                a_mets = new_plot[metric][author]
            except KeyError:
                a_mets = new_plot[metric]
                a_mets[author] = {}

            for k, v in vals.items():
                a_mets[author][k] = v

    return_values = {m: {} for m in metric_names}
    for metric in metric_names:
        temp = (
            pd.DataFrame.from_dict(new_plot[metric], orient="index")
            .reset_index(drop=False)
            .rename(columns={"index": "Author"})
        )
        assert len(temp), f"No plot data found for metric {metric}"
        cols = [c for c in temp.columns if c != "Author"]
        temp = temp.explode(column=cols).reset_index(drop=True)

        if metric == "DET":
            # return_values[f"{metric} (All Authors)"] = temp.to_dict()
            summary_formatted_for_plot = {}
            filt = temp["Author"] == "Summary"
            fpr = list(temp.loc[filt, "FPR"].astype(float))
            fnr = list(temp.loc[filt, "FNR"].astype(float))

            ticks = [0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99]
            tick_locations = list(stats.norm.ppf(ticks))
            tick_labels = [f"{s:.0%}" if (100 * s).is_integer() else f"{s:.1%}" for s in ticks]

            coords = [[x, y] for x, y in zip(np.array(fpr), np.array(fnr))]
            simplified = np.asarray(simplify_coords(coords, 0.0001))
            summary_formatted_for_plot["x_values"] = dict(zip(range(len(simplified)), list(simplified[:, 0])))
            summary_formatted_for_plot["y_values"] = dict(zip(range(len(simplified)), list(simplified[:, 1])))
            summary_formatted_for_plot["Author"] = dict(zip(range(len(simplified)), ["Summary"] * len(simplified)))

            summary_formatted_for_plot["x_label"] = "False Positive Rate"
            summary_formatted_for_plot["y_label"] = "False Negative Rate"
            summary_formatted_for_plot["x_limits"] = [-3.0, 3.0]
            summary_formatted_for_plot["y_limits"] = [-3.0, 3.0]
            summary_formatted_for_plot["x_tick_locations"] = tick_locations
            summary_formatted_for_plot["y_tick_locations"] = tick_locations
            summary_formatted_for_plot["x_tick_labels"] = tick_labels
            summary_formatted_for_plot["y_tick_labels"] = tick_labels

            summary_formatted_for_plot["x_locations"] = list(sp.stats.norm.ppf(list(simplified[:, 0])))
            summary_formatted_for_plot["y_locations"] = list(sp.stats.norm.ppf(list(simplified[:, 1])))
            if eer_to_transform is not None:
                summary_formatted_for_plot["Transformed Equal Error Rate"] = sp.stats.norm.ppf(eer_to_transform)

            return_values[metric] = summary_formatted_for_plot
        else:
            return_values[metric] = temp.to_dict()

    return return_values


def instance_attribution_metrics(prob, true, pred, far_target, which_metrics):
    """Calculate TA2 metrics for a particular instance."""
    author_mets = {}
    author_plts = {}
    tars = []

    if "auc" in which_metrics or "all" in which_metrics:
        area = auc(prob, true)
        author_mets.update({"Area Under ROC Curve": area})

    if "pauc" in which_metrics or "all" in which_metrics:
        partial_area = pauc(prob, true)
        author_mets.update({"partial Area Under ROC Curve": partial_area})

    if "precision" in which_metrics or "all" in which_metrics or "recall" in which_metrics or "f1" in which_metrics:
        precision, recall, f1, _ = prf(pred, true)
        author_mets.update({"Precision": precision, "Recall": recall, "F1": f1})
    if "det" in which_metrics or "all" in which_metrics:
        # TODO: Clean up the output format of TA2 plots
        fpr, fnr, thresh_det = det(prob, true)
        author_plts["DET"] = {"FPR": fpr, "FNR": fnr, "Threshold": thresh_det}

    far, tar, thresh = tar_at_far(prob, true)
    if "eer" in which_metrics or "all" in which_metrics:
        fpr, fnr, thresh_det = det(prob, true)
        eer_metric = eer(fpr, fnr)
        author_mets.update({"Equal Error Rate": eer_metric})

    if "dcf" in which_metrics or "all" in which_metrics:
        dcf_val = dcf(far, tar)
        author_mets.update({"Detection Cost Function": dcf_val})

    if "tar_at_far" in which_metrics or "all" in which_metrics:
        author_plts["Tar@Far"] = {"TAR": tar, "FAR": far}
        for f in far_target:
            t, far, tar, thresh = tar_at_far(prob, true, f)
            tars.append(t)

        author_mets.update({"TAR@" + str(f): t for f, t in zip(far_target, tars)})

    instance_ = pd.Series(author_mets)

    return instance_, author_plts


def prf(y_pred, y_true):
    """Compute precision, recall, and f1-score."""
    return precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average="binary", zero_division=0)


def auc(probs, true):
    """Build the AUROC."""
    return roc_auc_score(true.flatten(), probs.flatten())


def pauc(probs, true):
    """Compute a partial AUROC with an FPR range from 0 to 0.01."""
    return roc_auc_score(true.flatten(), probs.flatten(), max_fpr=0.01)


def tar_at_far(probs, true, far_target=None, return_thresh=False):
    """Compute the tar@far."""
    true_flat = true.flatten()
    probs_flat = probs.flatten()

    far, tar, thresh = roc_curve(true_flat, probs_flat)

    if far_target is not None:
        idx = np.abs(far - far_target).argmin()
        return tar[idx], far, tar, thresh
    else:
        return far, tar, thresh


def det(probs, true):
    """Build the DET curve."""
    true_flat = true.flatten()
    probs_flat = probs.flatten()

    fpr, fnr, thresh = det_curve(true_flat, probs_flat)
    return fpr, fnr, thresh


def eer(fpr, fnr):
    """Compute the EER."""
    idx = np.nanargmin(np.absolute(fnr - fpr))
    EER = np.mean([fpr[idx], fnr[idx]])

    return EER


def dcf(fprs, tprs, c_miss=10, c_fa=1, p_target=0.01):
    """Compute the dcf."""
    fnrs = 1 - tprs
    min_c_det = float("inf")
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            # min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf
