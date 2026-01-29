import numpy as np
from sklearn.preprocessing import StandardScaler


AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
AA_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}




def compute_aac(seq):
vec = np.zeros(20)
for aa in seq:
vec[AA_INDEX[aa]] += 1
return vec / len(seq)




def compute_dpc(seq):
vec = np.zeros((20, 20))
for i in range(len(seq) - 1):
a1, a2 = seq[i], seq[i + 1]
vec[AA_INDEX[a1], AA_INDEX[a2]] += 1
return vec.flatten() / (len(seq) - 1)

def extract_features(sequences):
features = []
for seq in sequences:
aac = compute_aac(seq)
dpc = compute_dpc(seq)
features.append(np.concatenate([aac, dpc]))
return np.array(features)




def scale_features(X_train, X_val, X_test):
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
return X_train, X_val, X_test
