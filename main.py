from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import tiktoken

app = Flask(__name__)
CORS(app)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load the NanoGPT model
class NanoGPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(50257, 64)
        self.lm_head = torch.nn.Linear(64, 50257)
        self.block_size = 16  # Must match training block_size

    def forward(self, x, targets=None):
        x = self.embed(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0):
        for _ in range(max_new_tokens):
            # Crop idx to block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Get last token's logits
            logits = logits[:, -1, :] / temperature
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize and load model
model = NanoGPT()
try:
    model.load_state_dict(torch.load("model1.pt", map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model loaded successfully from model1.pt")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = min(int(data.get("num_tokens", 50)), 100)  # Limit to 100 tokens

    try:
        # Tokenize input
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        
        # Generate text
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens)
        
        # Decode only new tokens
        new_tokens = output_ids[0, input_ids.shape[1]:].tolist()
        generated_text = tokenizer.decode(new_tokens)
        
        return jsonify({
            "generated_text": generated_text,
            "tokens_generated": len(new_tokens)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)