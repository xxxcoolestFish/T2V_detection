import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('/root/autodl-tmp/msc_eval_results.csv')
y_true = df['True'].values
y_prob = df['Prob'].values

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)

# roc_curve returns thresholds in DESCENDING order
# Sort ascending for searchsorted
sort_idx = np.argsort(thresholds)
thr_asc = thresholds[sort_idx]
fpr_asc = fpr[sort_idx]
tpr_asc = tpr[sort_idx]

targets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
annotations = []
for t in targets:
    idx = np.searchsorted(thr_asc, t)
    idx = min(idx, len(fpr_asc)-1)
    annotations.append((t, fpr_asc[idx], tpr_asc[idx]))

print('AUC: %.4f' % auc)
print()
print('%-12s %-10s %-14s %-10s %-12s %-10s' % ('Threshold', 'FPR', 'TPR(Recall)', 'TNR', 'Precision', 'F1'))
print('-' * 68)
n_real, n_fake = 1477, 523
for t, fp, tp in annotations:
    tn = 1 - fp
    tp_count = tp * n_fake
    fp_count = fp * n_real
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    f1 = 2 * tp * precision / (tp + precision) if (tp + precision) > 0 else 0
    print('%-12.1f %-10.4f %-14.4f %-10.4f %-12.4f %-10.4f' % (t, fp, tp, tn, precision, f1))

print()
print('--- Key FPR targets ---')
for target_fpr in [0.05, 0.10, 0.20]:
    idx = np.searchsorted(fpr_asc, target_fpr)
    idx = min(idx, len(fpr_asc)-1)
    print('FPR=%.2f: threshold=%.3f, TPR=%.4f' % (target_fpr, thr_asc[idx], tpr_asc[idx]))

print()
print('--- Per-source at threshold=0.3 ---')
for src in sorted(df['Source'].unique()):
    sub = df[df['Source'] == src]
    pred = (sub['Prob'] >= 0.3).astype(int)
    true = sub['True'].values
    acc = (pred == true).mean()
    if true[0] == 0:
        fp_rate = (pred == 1).mean()
        print('  %-18s Acc=%.1f%%  FP=%.1f%%  n=%d' % (src, acc*100, fp_rate*100, len(sub)))
    else:
        recall = (pred == 1).mean()
        print('  %-18s Acc=%.1f%%  Recall=%.1f%%  n=%d' % (src, acc*100, recall*100, len(sub)))

# Also check prob distribution
print()
print('--- Probability distribution ---')
for src in sorted(df['Source'].unique()):
    sub = df[df['Source'] == src]
    p = sub['Prob'].values
    print('  %-18s mean=%.3f median=%.3f min=%.3f max=%.3f' % (src, p.mean(), np.median(p), p.min(), p.max()))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(fpr, tpr, 'b-', linewidth=2, label='ROC (AUC=%.4f)' % auc)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
for t, fp, tp in annotations:
    ax.plot(fp, tp, 'ro', markersize=6)
    ax.annotate('t=%.1f' % t, (fp, tp), textcoords='offset points',
                xytext=(10, -5), fontsize=8, color='red')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('MSC v3 - ROC Curve with Threshold Annotations')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/msc_roc_curve.png', dpi=150)
print('\nPlot saved.')
