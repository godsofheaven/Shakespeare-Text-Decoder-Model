## Shakespeare GPT-2 (124M) — Training, Generation, and HF App

### Overview
This repository contains a compact GPT-2 style language model trained on Shakespeare’s works. It includes:
- Training script with state-of-the-art techniques (`src/train.py`)
- Generation script for inference (`src/generate.py`)
- Configurable hyperparameters (`configs/config.yaml`)
- Logged metrics in JSON and a human-friendly summary
- A Hugging Face Gradio app for deployment (`hf_app/`)

Final training achieved loss 0.094349 at step 1700.

### Directory Structure
- `src/` — core code
  - `models/gpt.py` — GPT-2 (decoder-only) model
  - `data/data_loader.py` — tiktoken-based loaders and utilities
  - `train.py` — training with warmup, cosine decay, grad accumulation, clipping, checkpoints, JSON logging
  - `generate.py` — CLI text generation
  - `checkpoints/` — saved checkpoints (best/final/intermediate)
- `configs/config.yaml` — model/training/data/system paths
- `data/input.txt` — Shakespeare corpus (sample provided)
- `logs/` — structured logs
  - `training_log.json` — intermediate JSON logs
  - `training_log_final.json` — final JSON logs (includes full curve)
  - `logs.txt` — human-friendly summary derived from JSON logs
- `hf_app/` — Gradio app for Hugging Face Spaces deployment

### Environment Setup
1) Create environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

2) Verify PyTorch device (CUDA/MPS/CPU) is detected:

```python
import torch; print(torch.cuda.is_available(), getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
```

### Data
- Default training file: `data/input.txt`
- To use your own corpus, replace `data/input.txt` or update paths in `configs/config.yaml`.

### Training
From the repo root:

```bash
python src/train.py
```

What it does:
- Uses GPT-2 architecture (12 layers, 12 heads, 768 dim; block size 1024)
- Warmup + cosine LR schedule, AdamW with parameter-specific weight decay
- Gradient accumulation (effective batch size = `batch_size * grad_accum_steps`)
- Gradient clipping and periodic checkpoints
- Structured JSON logging to `logs/training_log.json` and `logs/training_log_final.json`

Artifacts:
- Best checkpoint: `src/checkpoints/best_model.pt`
- Intermediate checkpoints: `src/checkpoints/checkpoint_step_*.pt`
- Final checkpoint (when target met): `src/checkpoints/final_step_XXXX_loss_*.pt`

Results (from logs):
- Final step: 1700
- Final loss: 0.094349 (target < 0.099999)

### Inference / Text Generation
Use any checkpoint (best or final). Example:

```bash
python src/generate.py --checkpoint src/checkpoints/best_model.pt --prompt "ROMEO:" --max-tokens 150 --temperature 0.8 --top-k 50
```

Interactive mode:

```bash
python src/generate.py --checkpoint src/checkpoints/best_model.pt --interactive
```

### Configuration
Tune hyperparameters and paths in `configs/config.yaml`:
- Model: `n_layer`, `n_head`, `n_embd`, `block_size`, `vocab_size`
- Training: `batch_size`, `sequence_length`, `max_steps`, `learning_rate`, `warmup_steps`, `weight_decay`, `grad_clip`, `log_interval`, `eval_interval`, `save_interval`, `target_loss`
- Data: `train_file`, `encoding`, `train_split`
- System: `device` ("auto" uses CUDA/MPS/CPU), `seed`
- Paths: `checkpoint_dir`, `log_dir`, `data_dir`

### Logging and Metrics
During training:
- JSON logs are written to:
  - `logs/training_log.json` (periodic)
  - `logs/training_log_final.json` (when target is met)
- A readable summary is saved to `logs/logs.txt` (generated from the JSON logs).

Quick visualization snippet:

```python
import json, matplotlib.pyplot as plt
with open("logs/training_log_final.json") as f:
    data = json.load(f)
steps  = [e["step"] for e in data["training_log"]]
losses = [e["loss"] for e in data["training_log"]]
plt.plot(steps, losses); plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss"); plt.grid(True); plt.show()
```

### Hugging Face Gradio App
The `hf_app/` directory contains a lightweight Gradio app:
- `hf_app/app.py` — UI and generation loop
- `hf_app/model_quantized.pt` — 330MB FP16 quantized checkpoint (for deployment)
- `hf_app/requirements.txt` — dependencies for Spaces
- `hf_app/README.md` — Space description
- `hf_app/UPLOAD_TO_HF.md` — step-by-step instructions to create and push a Space

Deploy summary:
1) Create a Gradio Space on Hugging Face.
2) Upload `app.py`, `model_quantized.pt`, `requirements.txt`, and `README.md` from `hf_app/`.
3) Ensure Git LFS is enabled for `.pt` files.

### Tips
- For faster training, lower `sequence_length` or `batch_size`, or train longer for lower loss.
- Use CUDA if available; otherwise CPU training will be much slower.
- Replace `data/input.txt` with your domain text to fine-tune the style.

### Screenshots
See the included screenshots in the repo root for sample results and UI previews.


