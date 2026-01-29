import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score




def evaluate(model, X_test, y_test):
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.eval()
with torch.no_grad():
logits = model(torch.tensor(X_test).float().to(device)).squeeze()
probs = torch.sigmoid(logits).cpu().numpy()
preds = (probs > 0.5).astype(int)


acc = accuracy_score(y_test, preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
auc = roc_auc_score(y_test, probs)


return acc, prec, rec, f1, auc
