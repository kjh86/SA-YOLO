# SA-YOLO
SA-YOLO (Stage-Aware YOLO) is an attention-centric real-time object detector designed for smoke and fire detection.
This repository provides the official implementation of SA-YOLO, built on top of a modified Ultralytics YOLO framework (v8.3.96).

Overview

Detecting smoke and fire in real-world surveillance scenarios is challenging due to:

small-scale flame regions,

low-contrast and diffused smoke patterns,

and large appearance variations across scenes.

To address these issues, SA-YOLO introduces a Stage-Aware Attention strategy, where different attention mechanisms are selectively applied to different feature stages according to their semantic roles.

Key Contributions

Stage-Aware Attention Design

Lightweight attention modules are selectively assigned to feature stages (P3–P5).

Efficient and Modular Architecture

All attention blocks are plug-and-play and preserve feature map dimensions.

Real-Time Performance

Designed to maintain real-time inference speed while improving detection accuracy.

Robustness under Degradation

Improved performance under smoke-only and fog-degraded conditions.

Model Architecture

SA-YOLO extends Ultralytics YOLO by integrating the following attention modules:

ECA: Efficient Channel Attention

ResECA: Residual variant of ECA

PAM: Parallel Attention Module

Stage-Aware Assignment:

Low-level stages emphasize local texture preservation.

High-level stages emphasize global context modeling.

The implementation modifies internal model parsing and module registration logic to support stage-aware attention.

Implementation Details

Base Framework: Ultralytics YOLO v8.3.96

Language: Python

Deep Learning Framework: PyTorch

License: AGPL-3.0 (Ultralytics) + Apache-2.0 (this repository)

This repository contains a modified version of Ultralytics YOLO, extended to support SA-YOLO modules.

⚙️ Installation

Clone the repository and install in editable mode:

git clone https://github.com/kjh86/SA-YOLO.git
cd SA-YOLO
pip install -e .


Tested with ultralytics==8.3.96, Python 3.9, and PyTorch ≥ 1.13.

Training

Example training command:

yolo detect train \
  model=configs/sa-yolo.yaml \
  data=your_dataset.yaml \
  epochs=100 \
  imgsz=640

Evaluation
yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=your_dataset.yaml

Repository Structure
SA-YOLO/
├── ultralytics/          # Modified Ultralytics YOLO source (v8.3.96)
├── configs/              # SA-YOLO model configuration files
├── README.md
├── LICENSE
└── .gitignore

Citation

If you find this work useful, please consider citing:

@article{kim2025sayolo,
  title   = {SA-YOLO: Stage-Aware Attention for Real-Time Smoke and Fire Detection},
  author  = {Kim, Jin},
  journal = {IEEE Access},
  year    = {2025}
}

License

This project is released for academic and research purposes.

Ultralytics YOLO components follow the AGPL-3.0 license.

SA-YOLO modifications are provided under Apache-2.0.

Contact

For questions or collaborations, feel free to open an issue or contact the author.
