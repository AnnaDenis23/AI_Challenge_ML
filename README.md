# AI Challenge — Computer Vision Solutions

**Author:** Anastasiya Denisenko  
**GitHub:** [AnnaDenis23](https://github.com/AnnaDenis23)  
**Competition:** AI Challenge 2026  

---

##  About This Repository

This repository contains my solutions to two **Computer Vision** tasks from the **AI Challenge** competition. Both tasks were solved using **Transfer Learning** with **ResNet18** pretrained on ImageNet.

The projects demonstrate practical application of deep learning for real-world problems: autonomous driving (weather classification) and video surveillance (lighting level classification).

| Project | Task | Metric | Best Result | Leaderboard |
|---------|------|--------|-------------|-------------|
| [Weather Classification](#-project-1-weather-classification) | Weather condition classification (rain/fog/snow) | Macro F1-score | **0.94** | **70 / 114** |
| [Light Level Classification](#-project-2-light-level-classification) | Lighting level classification (dark/normal/bright) | Accuracy | **0.52** | **40 / 140** |

---


**Note:** Full datasets are not included in this repository due to size limits.  
Download links are provided in each project section.

---


**Note:** Full datasets are not included in this repository due to size limits.  
Download links are provided in each project section.

---

##  Project 1: Light Level Classification

> **"Уровень освещённости" (Light Level Classification)** — Determine lighting conditions from camera images

###  Problem Statement

A student named Yaroslav is helping relatives set up a video surveillance system at their country house. After basic camera setup, he moved on to testing computer vision algorithms on video streams.

He noticed that object recognition quality significantly depends on scene illumination — different models perform differently under bright and dim lighting conditions.

Therefore, to select the appropriate model, he needs to **automatically determine the lighting level** of an image (classify by degree of illumination: low, medium, or high).

**Important:** Time of day cannot be used as a feature because cameras are located in different conditions and have different exposure settings.

**Input:** Images from surveillance cameras  
**Output:** One of 3 classes:
- `0` — dark (низкая освещенность)
- `1` — normal (средняя освещённость)
- `2` — bright (высокая освещённость)

**Evaluation Metric:** Accuracy  
**Passing Threshold:** Accuracy ≥ 0.40  
**Target Scores:** ≥ 0.70 (good), ≥ 0.90 (excellent)

---

###  My Approach

#### 1. Exploratory Data Analysis (EDA)

- **Dataset size:** ~2500 training images
- **Class distribution:** Imbalanced
  - Normal: 40% (1000 images)
  - Bright: 35% (875 images)
  - Dark: 25% (625 images)
- **Key challenges identified:**
  - Low-light images have high noise levels
  - Overexposed images lose detail in bright regions
  - Shadows and reflections can mislead the classifier
  - Lighting conditions vary across different camera settings
  - High intra-class variance (different times of day, weather conditions)

#### 2. Data Preprocessing

All images were resized to **224×224** and normalized using ImageNet statistics:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
---
##  Project 2: Weather Classification

> **"За окном шумно" (It's Noisy Outside)** — Weather condition classifier

###  Problem Statement

A student named Ivan won a tender to develop a **computer vision system for an autonomous driving company**. One of the key subtasks is determining weather conditions from camera images, as this directly impacts driving safety and control algorithms.

Images may contain various distortions — noise, blur, reduced contrast, and other effects caused by weather conditions (rain, fog, snow).

**Input:** Road scene images from car cameras  
**Output:** One of 3 classes:
- `0` — rain (дождь)
- `1` — fog (туман)
- `2` — snow (снег)

**Evaluation Metric:** Macro F1-score  
**Passing Threshold:** F1 ≥ 0.50  
**Target Scores:** ≥ 0.75 (good), ≥ 0.90 (excellent)

---

###  My Approach

#### 1. Exploratory Data Analysis (EDA)

- **Dataset size:** ~2000 training images
- **Class distribution:** Balanced (500–750 images per class)
- **Key challenges identified:**
  - Images contain noise, motion blur, and low contrast (especially foggy conditions)
  - Snow and rain can visually resemble each other in certain conditions
  - Lighting variations affect color perception

#### 2. Data Preprocessing

All images were resized to **224×224** and normalized using ImageNet statistics:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
