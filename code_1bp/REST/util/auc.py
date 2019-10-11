import numpy as np

def score_record(truth, predictions, input_digits=None):
    if input_digits is None: # bin resolution
        input_digits = 3 
    scale=10**input_digits
    pos_values = np.zeros(scale + 1, dtype=np.int64)
    neg_values = np.zeros(scale + 1, dtype=np.int64)
    b = scale+1
    r = (-0.5 / scale, 1.0 + 0.5 / scale)
    all_values = np.histogram(predictions, bins=b, range=r)[0]
    if np.sum(all_values) != len(predictions):
        raise ValueError("invalid values in 'predictions'")
    pred_pos = predictions[truth > 0]
    pos_values = np.histogram(pred_pos, bins=b, range=r)[0]
    pred_neg = predictions[truth == 0]
    neg_values = np.histogram(pred_neg, bins=b, range=r)[0]
    return (pos_values, neg_values)

def calculate_auc(pos_values,neg_values): # auc & auprc; adapted from score2018.py
    tp = np.sum(pos_values)
    fp = np.sum(neg_values)
    tn = fn = 0 
    tpr = 1 
    tnr = 0 
    if tp == 0 or fp == 0:
        # If either class is empty, scores are undefined.
        return (float('nan'), float('nan'))
    ppv = float(tp) / (tp + fp) 
    auroc = 0 
    auprc = 0 
    for (n_pos, n_neg) in zip(pos_values, neg_values):
        tp -= n_pos
        fn += n_pos
        fp -= n_neg
        tn += n_neg
        tpr_prev = tpr 
        tnr_prev = tnr 
        ppv_prev = ppv
        tpr = float(tp) / (tp + fn)
        tnr = float(tn) / (tn + fp)
        if tp + fp > 0:
            ppv = float(tp) / (tp + fp)
        else:
            ppv = ppv_prev
        auroc += (tpr_prev - tpr) * (tnr + tnr_prev) * 0.5
        auprc += (tpr_prev - tpr) * ppv_prev
    return (auroc, auprc)
