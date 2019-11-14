import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc


#
def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'best f1':
        return best_f1(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")

#
def best_f1(labels, scores):
    """ Evaluate the best F1 score

    Returns:
        best, acc, sens, spec: the best F1 score, accuracy, sensitivity, specificity
    """
    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    best = 0.0
    best_threshold = -1

    for threshold in thresholds[1:]:
        tmp_scores = scores.clone().detach()
        tmp_scores[tmp_scores >= threshold] = 1
        tmp_scores[tmp_scores <  threshold] = 0
        f1 = f1_score(labels, tmp_scores)
        if best < f1:
            best = f1
            best_threshold = threshold
    
    preds =  scores.clone().detach()
    preds[preds >= best_threshold] = 1
    preds[preds <  best_threshold] = 0

    TP = preds[labels == 1].sum().item() # True positive
    CP = (labels == 1).sum().item() # Condition positive = TP + FN
    TN = (labels == 0).sum().item() - preds[labels == 0].sum().item()
    CN = (labels == 0).sum().item()

    acc = (TP + TN) / (CP + CN)
    sens = TP / CP
    spec = TN / CN

    return best, acc, sens, spec, best_threshold

#
def roc(labels, scores, saveto='./'):
    """ Evaluate ROC

    Returns:
        auc, eer: Area under the curve, Equal Error Rate
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.png"))
        plt.close()

    return roc_auc
