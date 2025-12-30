# SA-YOLO (Stage-Aware YOLO)

SA-YOLO is an attention-centric real-time detector for **smoke and fire detection**.  
This repository provides an official implementation built on top of **Ultralytics YOLO (base: v8.3.96)** with stage-aware attention modules.

---

## Highlights
- **Stage-Aware Attention**: different lightweight attention modules are assigned to different feature stages (P3–P5)
- **Plug-and-Play Modules**: modules preserve feature map shape (drop-in replacement)
- **Real-time Friendly**: designed for practical surveillance scenarios
- **Robust under Degradation**: improved performance under smoke-only and fog-degraded conditions

---

## Quick Start

###1) Clone & Install
    git clone https://github.com/kjh86/SA-YOLO.git
    cd SA-YOLO
    pip install -e .

2) Verify Installation
    yolo help

Training / Evaluation

Update paths to your dataset YAML and model YAML.

Train
    yolo detect train \
    model=ultralytics/cfg/models/custom/yolov12-stage-aware.yaml \
    data=path/to/your_dataset.yaml \
    epochs=100 imgsz=640

Validate
    yolo detect val \
    model=runs/detect/train/weights/best.pt \
    data=path/to/your_dataset.yaml

What’s Inside (Key Modifications)
This repository includes:
Modified Ultralytics source (base v8.3.96) under ultralytics/
SA-YOLO attention modules
ECA (Efficient Channel Attention)
ResECA (Residual ECA)
PAM (Parallel Attention Module)
Model configs for SA-YOLO and ablations under:
ultralytics/cfg/models/custom/
Core changes are typically in:
ultralytics/nn/tasks.py (model parsing / module registration)
ultralytics/nn/modules/conv.py (custom modules or registry hooks)
