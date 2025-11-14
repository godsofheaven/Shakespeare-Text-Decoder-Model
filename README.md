Training Summary
====================================================

- Final Step: 1700
- Final Loss: 0.094349

Step-wise Metrics (every 100 steps)
-----------------------------------
- Step 0100 | Loss: 8.627984 | LR: 3.0000e-05 | Tokens/sec: 640,857.68 | Time/step: 0.025566 s
- Step 0200 | Loss: 6.139783 | LR: 6.0000e-05 | Tokens/sec: 730,760.44 | Time/step: 0.022420 s
- Step 0300 | Loss: 4.781182 | LR: 9.0000e-05 | Tokens/sec: 852,129.77 | Time/step: 0.019227 s
- Step 0400 | Loss: 4.050571 | LR: 1.2000e-04 | Tokens/sec: 749,582.52 | Time/step: 0.021858 s
- Step 0500 | Loss: 3.342267 | LR: 1.5000e-04 | Tokens/sec: 726,804.29 | Time/step: 0.022543 s
- Step 0600 | Loss: 2.671492 | LR: 1.8000e-04 | Tokens/sec: 825,389.68 | Time/step: 0.019850 s
- Step 0700 | Loss: 2.020290 | LR: 2.1000e-04 | Tokens/sec: 784,926.25 | Time/step: 0.020873 s
- Step 0800 | Loss: 1.354819 | LR: 2.4000e-04 | Tokens/sec: 791,411.83 | Time/step: 0.020702 s
- Step 0900 | Loss: 0.772771 | LR: 2.7000e-04 | Tokens/sec: 822,313.74 | Time/step: 0.019924 s
- Step 1000 | Loss: 0.398528 | LR: 3.0000e-04 | Tokens/sec: 856,510.44 | Time/step: 0.019129 s
- Step 1100 | Loss: 0.228469 | LR: 3.3000e-04 | Tokens/sec: 844,678.48 | Time/step: 0.019397 s
- Step 1200 | Loss: 0.160448 | LR: 3.6000e-04 | Tokens/sec: 771,079.59 | Time/step: 0.021248 s
- Step 1300 | Loss: 0.133567 | LR: 3.9000e-04 | Tokens/sec: 851,634.06 | Time/step: 0.019238 s
- Step 1400 | Loss: 0.124452 | LR: 4.2000e-04 | Tokens/sec: 862,335.76 | Time/step: 0.018999 s
- Step 1500 | Loss: 0.112699 | LR: 4.5000e-04 | Tokens/sec: 689,064.06 | Time/step: 0.023777 s
- Step 1600 | Loss: 0.101688 | LR: 4.8000e-04 | Tokens/sec: 880,089.76 | Time/step: 0.018616 s
- Step 1700 | Loss: 0.094349 | LR: 5.1000e-04 | Tokens/sec: 869,492.58 | Time/step: 0.018843 s

Notes
-----
- Target loss (< 0.099999) achieved at step 1700.
- Logs were generated via JSON during training and summarized here for quick reference.

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


