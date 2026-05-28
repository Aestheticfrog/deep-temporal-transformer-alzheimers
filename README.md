# 🧠 Deep Temporal Transformer for Alzheimer’s Disease Detection

> AI-powered early Alzheimer’s detection using MRI scans, Temporal Transformers, and Deep Learning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Transformers](https://img.shields.io/badge/Transformer-Architecture-purple)
![Medical AI](https://img.shields.io/badge/Domain-MedicalAI-green)
![Status](https://img.shields.io/badge/Status-ResearchProject-orange)

---

## 🚀 Overview

This project proposes a **Deep Temporal Transformer (DTT)** architecture for automated Alzheimer’s Disease detection using MRI scans.

Unlike traditional methods that analyze MRI slices independently, this system treats the entire MRI scan as a **patient-level temporal sequence**, enabling the model to learn long-range anatomical dependencies across brain slices.

The architecture combines:

- 🧠 **ResNet34** for spatial feature extraction
- 🔄 **Transformer Encoder** for temporal MRI sequence learning
- 🐻 **Brown Bear Optimization (BBO)** for feature-dimension tuning
- ⚡ **PyTorch** for high-performance deep learning implementation

The proposed framework achieved an **F1-score of 0.82** on binary dementia classification.

---

# 🔬 Research Motivation

Traditional machine learning approaches for Alzheimer’s detection often:
- depend heavily on handcrafted features
- process MRI slices independently
- lose inter-slice contextual information

This project addresses those limitations by introducing a **Deep Temporal Transformer** that captures both:
- spatial features within slices
- temporal relationships across slices

This creates a more biologically meaningful understanding of brain degeneration patterns.

---

# 🏗️ Architecture

## Pipeline

```text
MRI Scan Sequence
        ↓
ResNet34 Spatial Encoder
        ↓
Feature Projection Layer
        ↓
Temporal Progression Transformer
        ↓
Classification Head
        ↓
Demented / Non-Demented

dataset link :-

https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset
