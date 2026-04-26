# Federated Vision Transformer with Homomorphic Encryption for Privacy-Preserving Medical AI

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/Status-Research%20Project-green)

A federated learning framework for medical image classification under non-IID data distribution, integrating Vision Transformers (ViT) with CKKS-based homomorphic encryption to enable secure and privacy-preserving model aggregation.

---

## Overview

This project addresses the challenge of training deep learning models across multiple medical institutions without sharing sensitive patient data.

The system combines:

* Federated Learning (FL)
* Vision Transformers (ViT)
* Homomorphic Encryption (CKKS via TenSEAL)

to ensure **data privacy while maintaining high predictive performance**.

---

## Key Contributions

* Federated learning under **non-IID client distributions**
* Vision Transformer-based architecture for medical image classification
* Selective parameter sharing (**FeSViBS framework**)
* Integration of CKKS homomorphic encryption
* Secure aggregation under an honest-but-curious threat model

---

## System Architecture

![Architecture](https://raw.githubusercontent.com/Alishajindal/federated-learning-he/main/assets/architecture.png)

The system simulates distributed clients performing local training. Only encrypted model parameters are transmitted to a central server.

* No raw data exchange
* Encrypted parameter sharing
* Secure aggregation

---

## Dataset

* BloodMNIST (MedMNIST v2)
* 8-class classification problem
* 17,092 samples
* Non-IID distribution across 6 clients

---

## Results

### Model Performance

| Model                 | Balanced Accuracy |
| --------------------- | ----------------- |
| CNN (ResNet-18)       | 94.28%            |
| ViT (scratch)         | 88.45%            |
| ViT + HE (scratch)    | 82–83%            |
| ViT (pretrained)      | 94.91%            |
| ViT + HE (pretrained) | 94.41%            |

---

### Global Accuracy Trend

![Accuracy](https://raw.githubusercontent.com/Alishajindal/federated-learning-he/main/assets/accuracy.png)

---

### Confusion Matrix (Final Round)

![Confusion Matrix](https://raw.githubusercontent.com/Alishajindal/federated-learning-he/main/assets/confusion__matrix.png)

---

## System Evaluation

### Encryption Overhead

![Encryption Time](https://raw.githubusercontent.com/Alishajindal/federated-learning-he/main/assets/encryption_time.png)

* Average encryption time per client: ~65–73 seconds
* Homomorphic aggregation and decryption: ~69 seconds

---

## Key Insights

* Homomorphic encryption introduces **<1% accuracy degradation**
* Selective encryption significantly reduces computational overhead
* Stable convergence under non-IID data distribution
* Demonstrates feasibility of privacy-preserving AI in healthcare

---

## Repository Structure

```
src/        Federated learning pipeline
models/     Model architectures (ViT, FeSViBS)
he/         Homomorphic encryption utilities
app/        Streamlit-based inference interface
data/       Dataset handling and preprocessing
assets/     Visualization images for documentation
results/    Logs and evaluation outputs
```

---

## Technology Stack

* Python
* PyTorch
* Vision Transformers
* TenSEAL (CKKS Homomorphic Encryption)
* Streamlit

---

## Running the Project

```bash
pip install -r requirements.txt
python app/app.py
```

---

## Privacy and Security

* Secure aggregation using homomorphic encryption
* No exchange of raw client data
* Designed for cross-silo federated learning environments

---

## Future Work

* Deployment using Docker and Kubernetes
* Multi-node federated learning setup
* Optimization of encryption latency and memory usage

---

## Research Context

This work extends a capstone project into a research manuscript currently under review.

---

## Author

Alisha Jindal
B.E. Computer Engineering
Thapar Institute of Engineering and Technology

GitHub: https://github.com/Alishajindal
