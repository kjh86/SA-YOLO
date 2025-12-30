# SA-YOLO

**SA-YOLO (Stage-Aware YOLO)** is an attention-centric real-time object detector for **smoke and fire detection** in surveillance environments.  
This repository provides the **official implementation** of SA-YOLO built on top of a **modified Ultralytics YOLO (v8.3.96)** codebase.

---

## Highlights

- **Stage-Aware Attention (P3–P5)**: assigns different lightweight attention modules per feature stage based on semantic roles.
- **Plug-and-Play Modules**: attention blocks preserve feature map shape (easy integration and ablation).
- **Real-Time Oriented**: designed to retain real-time inference while improving detection quality.
- **Robustness**: improved performance under **smoke-only** and **fog-degraded** conditions.

---

## Motivation

Detecting smoke and fire in real-world CCTV footage is difficult due to:

- **Small-scale flames** (tiny, sparse regions)
- **Low-contrast / diffused smoke** (weak boundaries and global spread)
- **Large scene variations** (lighting, weather, camera distance, blur)

SA-YOLO addresses these issues with **stage-aware feature refinement**, applying attention differently across stages to better preserve **local textures** (low-level) and enhance **global context** (high-level).

---

## Method: Stage-Aware Attention

SA-YOLO integrates three lightweight attention blocks:

- **ECA**: Efficient Channel Attention  
- **ResECA**: Residual ECA variant  
- **PAM**: Parallel Attention Module  

**Stage-aware assignment (conceptual):**
- **Lower-level stages (e.g., P3)** → emphasize **local texture / edge preservation**
- **Higher-level stages (e.g., P5)** → emphasize **global context / semantic modeling**

> Internally, this implementation extends Ultralytics model parsing + module registration so stage-aware attention can be inserted cleanly and reproduced consistently.

---

## Environment

- **Base framework**: Ultralytics YOLO `v8.3.96`
- **Language**: Python
- **DL framework**: PyTorch
- Tested with: **Python 3.9**, **PyTorch ≥ 1.13**

---

## Installation

Clone and install in editable mode:

```bash
git clone https://github.com/kjh86/SA-YOLO.git
cd SA-YOLO
pip install -e .
