# Information Stability for Non-Invasive Reliability Monitoring of CNNs

This repository contains the code and experimental framework for studying **non-invasive information stability** as a reliability signal for convolutional neural networks (CNNs).
The project systematically evaluates the strengths and limitations of stability-based monitoring under **distribution shift, selective rejection, and adaptive adversarial attacks**.

The work builds on and extends ideas from *Entropy-Based Non-Invasive Reliability Monitoring of CNNs* , with a focus on **honest evaluation under modern threat models**.

---

## Motivation

Deep neural networks often fail silently when exposed to:

* out-of-distribution (OOD) inputs,
* domain shift,
* adversarial perturbations.

Many existing solutions require:

* retraining,
* architectural modification,
* or expensive inference-time procedures.

This project explores whether **information stability**, measured through internal activation perturbations, can serve as a **post-hoc, non-invasive reliability signal**—and, crucially, **where it fundamentally breaks down**.

---

## Key Findings

**What works**

* Information stability is highly effective for **semantic OOD detection** (e.g., CIFAR-10 vs SVHN).
* Stability enables **selective rejection** that removes OOD samples while preserving in-distribution accuracy.
* No retraining or model modification is required.

**What does not work**

* Stability-based methods **fail under fully adaptive white-box attacks** (FGSM, PGD, AutoAttack).
* Hybrid and directional variants do not recover robustness.
* Stability does **not correlate with correctness** on clean in-distribution data.

**Core conclusion**

> Non-invasive information stability is a strong *reliability and distribution-shift signal*, but it cannot guarantee adversarial robustness under adaptive threat models.

---

## Repository Structure

```
info_stable_mobilenet/
├── models/
│   └── mobilenet_wrapper.py        # MobileNet with activation hooks
│
├── detection/
│   ├── per_sample_score.py         # Random information stability
│   ├── directional_score.py        # Gradient-based directional stability
│   └── hybrid_score.py             # Hybrid random + directional stability
│
├── losses/
│   └── info_stability_loss.py      # Online stability regularization (training-time)
│
├── adv_attacks/
│   ├── fgsm.py
│   └── pgd.py
│
├── attacks/
│   └── autoattack_eval.py          # AutoAttack integration
│
├── training/
│   └── train_info_stable.py        # MobileNet training with optional stability loss
│
├── evaluation/
│   ├── eval_adversarial_detection.py
│   ├── eval_autoattack_detection.py
│   ├── eval_ood_detection.py
│   ├── eval_selective_rejection.py
│   └── eval_ood_rejection.py
│
├── data/
│   └── (CIFAR-10 / SVHN / CIFAR-100)
│
└── README.md
```

---

## Setup

### Environment

* Python ≥ 3.10
* PyTorch (GPU recommended)
* torchvision
* scikit-learn
* foolbox
* auto-attack

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Train a MobileNet-V2 classifier on CIFAR-10:

```bash
python -m training.train_info_stable
```

This produces:

```
mobilenet_info_stable.pth
```

Training supports **optional online information stability regularization**, though all core results are **post-hoc and non-invasive**.

---

## Experiments

### 1. Adversarial Detection

Evaluate stability under FGSM and PGD:

```bash
python -m evaluation.eval_adversarial_detection
```

Evaluate under AutoAttack:

```bash
python -m evaluation.eval_autoattack_detection
```

Result: detection collapses to near-random under adaptive white-box attacks.

---

### 2. OOD Detection

Evaluate semantic distribution shift:

```bash
python -m evaluation.eval_ood_detection
```

Typical results:

* SVHN AUROC ≈ 0.9+
* CIFAR-100 ≈ random (label-space shift, not distributional)
* Gaussian noise trivially detected

---

### 3. Selective Rejection (Risk-Aware Prediction)

Clean in-distribution rejection:

```bash
python -m evaluation.eval_selective_rejection
```

OOD selective rejection:

```bash
python -m evaluation.eval_ood_rejection
```

Key result:

> Rejecting ~5–10% of unstable samples removes ~99% of OOD inputs while preserving ID accuracy.

---

## Threat Model

* **White-box adaptive adversary**
* Full access to:

  * model parameters,
  * architecture,
  * detector logic,
  * gradients

This is a **deliberately strict** evaluation setting.

---

## Main Results

### Summary of Empirical Findings

| Task                            | Setting                   | Metric                   | Result      | Interpretation                                    |
| ------------------------------- | ------------------------- | ------------------------ | ----------- | ------------------------------------------------- |
| **OOD Detection**               | CIFAR-10 → SVHN           | AUROC                    | **0.92**    | Strong separation for semantic distribution shift |
| **OOD Detection**               | CIFAR-10 → CIFAR-100      | AUROC                    | ~0.47       | Label-space shift without distributional change   |
| **OOD Detection**               | CIFAR-10 → Gaussian Noise | AUROC                    | ~1.00*      | Trivially detected via extreme instability        |
| **Selective Rejection (OOD)**   | CIFAR-10 + SVHN           | OOD Rejected @ 5% Reject | **≈ 99%**   | Almost all OOD removed                            |
| **Selective Rejection (OOD)**   | CIFAR-10 + SVHN           | ID Accuracy              | **No drop** | In-distribution performance preserved             |
| **Selective Rejection (Clean)** | CIFAR-10 only             | Accuracy Gain            | None        | Stability ≠ confidence on clean data              |
| **Adversarial Detection**       | FGSM (white-box)          | AUROC                    | ~0.51       | Near-random                                       |
| **Adversarial Detection**       | PGD (white-box)           | AUROC                    | ~0.51       | Near-random                                       |
| **Adversarial Detection**       | AutoAttack (adaptive)     | AUROC                    | ~0.50       | Complete detector collapse                        |

* AUROC polarity is inverted for Gaussian noise due to score saturation; separation is trivial.

---

### Key Takeaways

* **Information stability is effective for semantic OOD detection**, especially when combined with selective rejection.
* **Stability enables risk-aware prediction**, filtering almost all OOD inputs while preserving ID accuracy.
* **Stability-based detectors fundamentally fail under fully adaptive white-box adversaries**, even when hybridized.
* **Non-invasive reliability ≠ adversarial robustness**.

---

### Conclusion

> **Information stability is a powerful non-invasive reliability signal for distribution shift and selective rejection, but it cannot guarantee robustness against adaptive adversarial attacks.**

---

## Research Positioning

This repository intentionally reports:

* **positive results** where stability is appropriate (OOD, rejection),
* **negative results** where it fails (adaptive attacks).

The goal is **clarity, not inflated robustness claims**.

---

## Citation

If you use this code or build on this work, please cite:

```
@article{entropy_stability_cnn,
  title={Entropy-Based Non-Invasive Reliability Monitoring of Convolutional Neural Networks},
  author={Nazeri, Amir and Hafez, Wael},
  year={2024}
}
```

And reference this repository for extended experiments on OOD detection, selective rejection, and AutoAttack evaluation.

---

## License

This project is intended for **research and academic use**.
Please check individual dependencies for their respective licenses.

---

## Final Note

This repository demonstrates an important distinction:

> **Reliability monitoring is not the same as adversarial robustness.**

Understanding that difference is the main contribution.

---
