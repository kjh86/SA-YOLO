# SA-YOLO

**SA-YOLO (Stage-Aware YOLO)** is an attention-centric real-time object detector for **smoke and fire detection** in surveillance environments.  
This repository provides the **reference implementation** of SA-YOLO built on top of a **modified Ultralytics YOLO (v8.3.96)** codebase.

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

## Installation

### 1) Clone and install in editable mode

```bash
git clone https://github.com/kjh86/SA-YOLO.git
cd SA-YOLO
pip install -e .
```

---

### 2) Verify installation

```bash
python -c "import ultralytics; print('ultralytics import OK')"
yolo -v
```

---

## Quick Start

### 3) Training

```bash
yolo detect train \
  model=ultralytics/cfg/models/custom/yolov12-stage-aware.yaml \
  data=your_dataset.yaml \
  epochs=100 \
  imgsz=640
```

---

### 4) Evaluation

```bash
yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=your_dataset.yaml
```

---

### 5) Inference

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=your_video_or_images
```

---

## Environment

```text
Python       >= 3.9
PyTorch      >= 1.13
Ultralytics  8.3.96 (modified)
```

---

## Repository Structure

```text
SA-YOLO/
├── ultralytics/                      # Modified Ultralytics YOLO source (v8.3.96)
│   ├── nn/
│   │   ├── tasks.py                  # Modified model parsing / registration
│   │   └── modules/conv.py           # Custom attention modules
│   └── cfg/models/custom/            # SA-YOLO model YAMLs
├── README.md
├── LICENSE
└── .gitignore
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{kim2025sayolo,
  title   = {SA-YOLO: Stage-Aware Attention for Real-Time Smoke and Fire Detection},
  author  = {Kim, Jin},
  journal = {IEEE Access},
  year    = {2025}
}
```

---

## License

This repository includes a **modified copy of Ultralytics YOLO**, which is licensed under the  
**GNU Affero General Public License v3.0 (AGPL-3.0)**.

### License scope

- The following files are **derivative works of Ultralytics YOLO** and are therefore licensed under **AGPL-3.0**:
  - `ultralytics/nn/tasks.py`
  - `ultralytics/nn/modules/conv.py`

- The following components are **original contributions for SA-YOLO** and are licensed under **Apache-2.0**:
  - SA-YOLO model configuration files (`ultralytics/cfg/models/custom/*.yaml`)
  - Documentation files (e.g., `README.md`)
  - Other standalone files that are not derived from Ultralytics source code

### Important note

Any file that is copied from, modified from, or tightly integrated with Ultralytics YOLO code
is considered a **derivative work** and remains subject to **AGPL-3.0**, regardless of additional
original contributions.

If you plan to use this repository in a **product or network-accessible service**, please ensure
that you fully understand and comply with the obligations of **AGPL-3.0**, including source
availability requirements.

See the `LICENSE` file for the full license text.

---

## Contact

For questions or collaborations, please open an issue on GitHub.


