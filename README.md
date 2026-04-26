# Federated Vision Transformer with Homomorphic Encryption for Privacy-Preserving Medical AI

This project presents a federated learning framework for medical image classification under non-IID data distributions, integrating Vision Transformers with CKKS-based homomorphic encryption to enable secure model aggregation without sharing raw data.

---

## Research Context

This work extends a capstone project into a research manuscript currently under review at *Elsevier (Future Generation Computer Systems)*.

---

## Key Contributions

* Federated learning under non-IID client distributions
* Vision Transformer (ViT) architecture for medical image classification
* Selective parameter sharing (FeSViBS framework)
* Integration of CKKS-based homomorphic encryption using TenSEAL
* Secure aggregation under an honest-but-curious threat model

---

## System Architecture

![Architecture](results/plots/architecture2.png)

The system simulates multiple distributed clients performing local training. Only encrypted model parameters (selected transformer components) are transmitted to the central server for aggregation.

* No raw data exchange between clients
* Encrypted parameter transmission
* Centralized secure aggregation

---

## Dataset

* BloodMNIST (MedMNIST v2)
* 8-class blood cell classification
* 17,092 samples
* Non-IID partitioning across 6 clients

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

### Training Behavior

#### Client Accuracy

![Client Accuracy](results/plots/client__acc.png)

#### Client Loss

![Client Loss](results/plots/client__loss.png)

---

### Confusion Matrix (Final Round)

![Confusion Matrix](results/plots/confusion__matrix.png)

The strong diagonal structure indicates consistent classification performance across all classes.

---

## System Evaluation

### Encryption Overhead

![Encryption Time](results/plots/enc_time.png)

* Average encryption time per client: approximately 65–73 seconds
* Homomorphic aggregation and decryption: approximately 69 seconds

---

### Memory Utilization

#### CPU Memory

![CPU Memory](results/plots/cpu_memory.png)

#### GPU Memory

![GPU Memory](results/plots/gpu_memory.png)

* CPU memory usage: approximately 15–16 GB
* GPU memory usage: approximately 6–8 GB

---

## Global Metrics

![Balanced Accuracy](results/plots/global_balanced_acc.png)

![Precision Recall F1](results/plots/global_prf.png)

The model demonstrates stable convergence across rounds, improving from approximately 0.68 to 0.81 balanced accuracy.

---

## Key Insights

* Homomorphic encryption introduces less than 1% degradation in accuracy
* Selective parameter encryption significantly reduces computational overhead
* The system achieves stable convergence under non-IID conditions
* Demonstrates feasibility of privacy-preserving federated learning in healthcare scenarios

---

## Project Structure

```bash
src/        # Federated learning pipeline
models/     # Model architectures (ViT, FeSViBS)
he/         # Homomorphic encryption utilities
app/        # Streamlit interface and inference
data/       # Dataset handling and preprocessing
results/    # Logs and evaluation plots
```

---

## Technology Stack

* Python
* PyTorch
* Vision Transformers
* TenSEAL (CKKS Homomorphic Encryption)
* Streamlit
* Git and Git LFS

---

## How to Run

```bash
pip install -r requirements.txt
python app/streamlit_app.py
```

---

## Privacy and Security

* Secure aggregation using homomorphic encryption
* Honest-but-curious adversarial model
* No exchange of raw client data

---

## Future Work

* Containerized deployment using Docker and Kubernetes
* Multi-node federated training setup
* Optimization of encryption latency and memory usage

---

## Author

Alisha Jindal
B.E. Computer Engineering
Thapar Institute of Engineering and Technology

GitHub: https://github.com/Alishajindal
