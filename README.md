🧠 Deep Temporal Transformer for Alzheimer’s Disease Detection

AI-powered Alzheimer’s detection using MRI temporal progression modeling, Transformers, and Deep Learning.

⸻

🚀 Overview

This project proposes a Deep Temporal Transformer (DTT) architecture for automated Alzheimer’s Disease detection using MRI scan progression data.

Unlike traditional CNN-based approaches that process MRI images independently, this system models MRI data as a temporal sequence across timestamps, enabling the network to capture long-range progression patterns and contextual dependencies throughout the brain structure.

The framework combines:

* 🧠 ResNet34 for spatial feature extraction
* 🔄 Transformer Encoder for temporal progression learning
* 🐻 Brown Bear Optimization (BBO) for feature-dimension tuning
* ⚡ PyTorch for scalable deep learning implementation

The proposed architecture achieved an F1-score of 0.82 on binary dementia classification.

⸻

🔬 Research Motivation

Most conventional Alzheimer’s detection systems:

* rely heavily on handcrafted features
* process MRI scans independently
* fail to capture temporal progression relationships

This project addresses these limitations through a Deep Temporal Transformer that learns:

* spatial representations within MRI scans
* temporal dependencies across timestamps
* progression-aware disease patterns

This enables a more clinically meaningful representation of neurodegenerative progression.

⸻

🏗️ Architecture Pipeline

MRI Temporal Sequence
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

⸻

🧠 Key Features

✅ Temporal progression modeling
✅ Transformer-based attention learning
✅ ResNet34 spatial encoding
✅ Brown Bear Optimization (BBO)
✅ Patient-level sequence learning
✅ Reduced data leakage with stratified validation
✅ Research-oriented Medical AI implementation

⸻

📊 Results

Metric	Score
F1 Macro	0.82
Task	Binary Dementia Classification
Framework	PyTorch
Optimizer	Adam
Validation	Patient-Level Stratified K-Fold

⸻

🛠️ Tech Stack

* Python
* PyTorch
* Transformers
* ResNet34
* NumPy
* Scikit-learn
* Matplotlib
* Google Colab

⸻

📚 Dataset

Dataset used:

https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset

⸻

⚙️ How To Run

Clone Repository

git clone https://github.com/Aestheticfrog/deep-temporal-transformer-alzheimers.git

Install Dependencies

pip install -r requirements.txt

Run Training

python train.py

⸻

📄 Research Paper

Research paper included in repository.

⸻

🎯 Future Improvements

* Multi-class dementia stage classification
* Vision Transformers (ViT)
* Explainable AI integration
* 3D temporal MRI modeling
* Clinical deployment pipeline

⭐ Why This Project Matters

This project demonstrates:

* Deep Learning expertise
* Transformer architecture understanding
* Temporal sequence modeling
* Medical imaging knowledge
* Research-oriented engineering
* Optimization algorithm implementation
* Real-world AI application develop
