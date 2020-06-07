import scipy.stats
import numpy as np
from sklearn import metrics


precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
auPR = metrics.auc(recall, precision)
