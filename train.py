import torch
from torch.utils.data import DataLoader, TensorDataset
from model import ProteinMLP




def train_model(X_train, y_train, X_val, y_val, epochs=40):
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ProteinMLP().to(device)


criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())


train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)


best_val_loss = float('inf')
patience, counter = 5, 0

for epoch in range(epochs):
model.train()
for xb, yb in train_loader:
xb, yb = xb.to(device), yb.to(device)
optimizer.zero_grad()
preds = model(xb).squeeze()
loss = criterion(preds, yb)
loss.backward()
optimizer.step()


model.eval()
val_loss = 0
with torch.no_grad():
for xb, yb in val_loader:
xb, yb = xb.to(device), yb.to(device)
preds = model(xb).squeeze()
val_loss += criterion(preds, yb).item()


val_loss /= len(val_loader)
if val_loss < best_val_loss:
best_val_loss = val_loss
counter = 0
else:
counter += 1
if counter >= patience:
break


return model
