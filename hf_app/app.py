"""
Shakespeare Text Generator - Hugging Face Gradio App
Trained GPT-2 model (124M params)
"""

import gradio as gr
import torch
import tiktoken
import os
from dataclasses import dataclass


# GPT Model Architecture
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


import torch.nn as nn
from torch.nn import functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Load model
print("Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig()
model = GPT(config)

# Load checkpoint
checkpoint_path = "model_quantized.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded quantized model from {checkpoint_path}")
    print(f"  Training loss: {checkpoint.get('loss', 'N/A')}")
    print(f"  Model size: 330MB (FP16 quantized)")
else:
    print("⚠️ Checkpoint not found. Please upload 'model_quantized.pt'")

model.to(device)
model.eval()
print(f"✓ Model ready on {device}")

# Tokenizer
enc = tiktoken.get_encoding('gpt2')


# ---- Derived Stats (dynamic, for UI) ----
try:
    model_params = sum(p.numel() for p in model.parameters())
    model_params_m = model_params / 1e6
except Exception:
    model_params = None
    model_params_m = None

training_loss = None
training_step = None
if 'checkpoint' in locals():
    training_loss = checkpoint.get('loss', None)
    training_step = checkpoint.get('step', None)

def build_stats_md() -> str:
    params_line = f"- **Parameters**: {model_params:,} ({model_params_m:.0f}M)" if model_params is not None else "- **Parameters**: 124M"
    loss_line = f"- **Training Loss**: {training_loss:.6f}" if isinstance(training_loss, (float, int)) else "- **Training Loss**: N/A"
    step_line = f"- **Training Step**: {training_step}" if training_step is not None else "- **Training Step**: N/A"
    return f"""
### 📊 Model Details
{params_line}
- **Architecture**: GPT-2 (Decoder-only Transformer)
{loss_line}
{step_line}
- **Model Format**: FP16 quantized (≈330MB)
- **Device**: {device.upper()}
""".strip()


def generate_text(prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt"""
    
    if not prompt:
        return "⚠️ Please enter a prompt!"
    
    try:
        # Tokenize
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k if (top_k and int(top_k) > 0) else None
            )
        
        # Decode
        generated_text = enc.decode(generated[0].tolist())
        
        return generated_text
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Example prompts
examples = [
    ["First Citizen:", 150, 0.8, 50],
    ["ROMEO:", 150, 0.8, 50],
    ["To be, or not to be,", 200, 0.7, 40],
    ["What light through yonder window breaks?", 150, 0.8, 50],
    ["Friends, Romans, countrymen,", 150, 0.8, 50],
]


# Gradio Interface with Teal Theme
with gr.Blocks(
    title="Shakespeare Text Generator",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="cyan",
        neutral_hue="slate"
    ),
    css="""
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3) !important;
    }
    h1 {
        color: #0f766e !important;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px 6px 0 0;
        border-radius: 8px;
        background: #ecfeff;
        color: #0f766e;
        font-size: 12px;
        border: 1px solid #ccfbf1;
    }
    """
) as demo:
    gr.Markdown(f"""
    # 🎭 Shakespeare Text Generator
    
    <div>
      <span class="badge">Architecture: GPT-2</span>
      <span class="badge">Device: {device.upper()}</span>
      <span class="badge">Quantized: FP16</span>
      <span class="badge">Params: {int(model_params_m):d}M</span>
    </div>
    
    Enter a Shakespearean prompt and watch the AI continue the text!
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter a Shakespearean prompt (e.g., 'First Citizen:', 'ROMEO:', 'To be, or not to be,')",
                lines=3
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=600,
                        value=150,
                        step=10,
                        label="Max Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        value=0.8,
                        step=0.1,
                        label="Temperature (creativity)"
                    )
                    top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=10,
                        label="Top-K (diversity) (0 disables)"
                    )
            
            generate_btn = gr.Button("✨ Generate Shakespeare", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="Generated Text",
                lines=15,
                show_copy_button=True
            )

            gr.Markdown(build_stats_md())
    
    gr.Markdown("""
    ### 💡 Tips:
    - **Temperature**: Lower (0.5-0.7) = more focused, Higher (0.9-1.2) = more creative
    - **Top-K**: Controls vocabulary diversity (40-60 recommended)
    - **Prompts**: Try character names (ROMEO:, JULIET:) or famous phrases
    """)
    
    gr.Examples(
        examples=examples,
        inputs=[prompt_input, max_tokens, temperature, top_k],
        label="Example Prompts"
    )
    
    gr.Markdown(build_stats_md())
    
    # Connect button
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens, temperature, top_k],
        outputs=output_text
    )


if __name__ == "__main__":
    demo.launch()

