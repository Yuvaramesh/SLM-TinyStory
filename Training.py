# train_gpt.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from model import GPT, GPTConfig

# === Hyperparameters ===
block_size = 256
batch_size = 64
learning_rate = 1e-3
max_iters = 50
eval_interval = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Dataset Class ===
class TinyStoriesDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+1+block_size], dtype=torch.long)
        return x, y

# === Load Data ===
train_data = TinyStoriesDataset('train.bin', block_size)
val_data = TinyStoriesDataset('val.bin', block_size)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

# === Model ===
config = GPTConfig(
    block_size=block_size,
    vocab_size=50257,
    n_layer=2,
    n_head=2,
    n_embd=64,
    dropout=0.1
)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# === Evaluation Function ===
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            break  # quick estimate from one batch
    model.train()
    return total_loss

# === Training ===
print("🚀 Starting training...")
total_start_time = time.time()
model.train()

for step in range(1, max_iters + 1):
    step_start = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break  # one batch per step

    # Logging step time and loss
    print(f"🟢 Step {step:>3} | Loss: {loss.item():.4f} | Time: {time.time() - step_start:.2f} sec")

    # Evaluation step
    if step % eval_interval == 0:
        val_loss = evaluate(model, val_loader)
        print(f"🔍 Eval @ Step {step:>3} | Val Loss: {val_loss:.4f}")

# === Save Model ===
torch.save(model.state_dict(), 'gpt_model_trained1.pt')
print(f"\n✅ Model saved as gpt_model_trained1.pt")
print(f"⏱️ Total training time: {time.time() - total_start_time:.2f} sec")
