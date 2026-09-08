# AI Challenge — Computer Vision Projects

**Author:**  Anastasiya Denisenko
**GitHub:** [AnnaDenis23](https://github.com/AnnaDenis23)

---

##  About This Repository

This repository contains my solutions to two **Computer Vision** tasks from the **AI Challenge** competition. Both tasks were solved using **Transfer Learning** with **ResNet18** pretrained on ImageNet.

| Project | Task | Metric | Best Result |
|---------|------|--------|-------------|
| [ Weather Classification](#-project-1-weather-classification) | Weather condition classification (rain/fog/snow) | Macro F1-score | **0.94** |
| [ Light Level Classification](#-project-2-light-level-classification) | Lighting level classification (dark/normal/bright) | Accuracy | **0.52** |

---

##  Project 1: Weather Classification

###  Problem Statement

The task **"За окном шумно" (It's Noisy Outside)** requires building a computer vision model to classify weather conditions from car camera images. This is a critical component for autonomous driving systems, as weather directly impacts driving safety and control algorithms.

**Input:** Road scene images  
**Output:** One of 3 classes:
- `0` — rain
- `1` — fog
- `2` — snow

**Evaluation Metric:** Macro F1-score  
**Passing Threshold:** F1 ≥ 0.50  
**Target:** ≥ 0.75 (good), ≥ 0.90 (excellent)

---

###  My Approach

#### 1. Exploratory Data Analysis (EDA)

- **Dataset size:** ~2000 images
- **Class distribution:** Balanced (500–750 images per class)
- **Key challenge:** Images contain noise, blur, low contrast (especially fog)
- **Conclusion:** A robust pretrained model with noise tolerance is required

#### 2. Data Preprocessing

All images were resized to **224×224** and normalized using ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]