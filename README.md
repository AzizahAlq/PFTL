# Personalized Federated Transfer Learning (PFTL)

Official implementation of the paper:

**Personalized Federated Transfer Learning for Intrusion Detection across Networks with Heterogeneous Feature Spaces, Model Architectures, and Label Spaces**

**Azizah Alqahtani, Walid Aljoby, Mohamed Ragab, Bouziane Brik, Muhamad Felamban, and Tarek Helmy**

---

## Overview

This repository provides the implementation and experimental framework for **Personalized Federated Transfer Learning (PFTL)**, a federated learning approach designed for intrusion detection across heterogeneous network environments.

Traditional Federated Learning (FL) generally relies on structural compatibility across participating clients, particularly compatible input representations and model architectures. These assumptions are difficult to maintain in realistic intrusion-detection environments, where organizations may use different traffic features, preprocessing pipelines, local models, and attack taxonomies.

PFTL addresses this challenge by enabling collaborative learning across clients with heterogeneous:

* **feature spaces**,
* **model architectures**,
* **label spaces**, and
* **non-IID data distributions**.

The central idea of PFTL is to restrict federation to a **compact shared intermediate representation layer**. Each client retains its feature extractor, adapter, and classifier head locally, while only the parameters of the shared layer are synchronized.

This creates a lightweight common interface for knowledge transfer without requiring identical raw feature dimensions, identical private architectures, or identical classifier heads.

---

## Key Idea

Instead of aggregating the complete client model, PFTL decomposes each local model into private and shared components:

```text
Heterogeneous Local Input
          │
          ▼
┌──────────────────────────┐
│ Private Feature Encoder  │
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│     Private Adapter      │
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│ Compact Shared Layer     │
│        Dense(q)          │
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│ Private Classifier Head  │
└──────────────────────────┘
          │
          ▼
    Local Prediction
```

Only the **compact shared layer** is communicated between clients and the central aggregator.

The following components remain local:

* raw client data,
* private feature encoder,
* private adapter,
* private classifier head.

Clients therefore need to agree only on the dimensionality of the compact shared interface rather than on their complete input, model, or output structures.

---

## Main Contributions

### 1. Unified PFTL Framework for Heterogeneous Intrusion Detection

PFTL enables collaborative intrusion detection across clients with simultaneous heterogeneity in:

* feature spaces,
* model architectures,
* label spaces, and
* local data distributions.

Rather than requiring complete model compatibility, PFTL establishes a compact common interface through which heterogeneous clients can exchange knowledge.

### 2. Representation-Level Knowledge Sharing

PFTL synchronizes only a **compact intermediate representation layer**.

The private feature extractor, adapter, and classifier head remain client-specific.

This allows clients to use different input feature dimensions and different private model structures while still participating in federated training.

### 3. Personalized Local–Global Knowledge Transfer

PFTL introduces a **γ-blending mechanism** that combines locally updated and globally aggregated shared-layer parameters.

For client \(i\):

```text
W_mixed = γ_local W_local + γ_global W_global
```

where:

```text
γ_local + γ_global = 1
```

The blending coefficients determine the balance between:

* **local specialization**, and
* **collaborative global knowledge**.

### 4. Adaptive Validation-Driven Personalization

In the heterogeneous multi-class setting, PFTL dynamically adapts the global blending coefficient according to each client's validation performance.

If collaborative knowledge improves local validation performance, the influence of the global representation can increase.

If the transferred representation is less beneficial, the client shifts toward its locally learned representation.

### 5. Validation-Based Safety Mechanism

PFTL incorporates a lightweight validation-based mechanism to reduce harmful transfer.

After constructing the blended shared representation, each client compares its validation Macro-F1 against the locally updated representation.

The transferred representation is accepted only when:

```text
F1_mixed ≥ F1_local + ε
```

Otherwise, the client retains the locally updated shared parameters.

This allows each client to regulate transferred knowledge according to its own validation behavior.

### 6. Communication-Efficient Knowledge Transfer

Because PFTL communicates only the compact shared layer rather than the complete model, the communication payload remains small.

The measured serialized communication cost is approximately:

| Setting     | Shared Dimension | Communication per Client per Round |
| ----------- | ---------------: | ---------------------------------: |
| Binary      |          `q = 4` |                         ~606 bytes |
| Multi-class |          `q = 8` |                       ~1,404 bytes |

All remaining client-specific model parameters stay local.

### 7. Evaluation under Controlled and Heterogeneous Settings

PFTL is evaluated using a two-phase experimental design.

The first phase provides a controlled binary setting for comparison with conventional federated-learning approaches.

The second phase evaluates the framework under heterogeneous multi-class conditions where clients differ in feature spaces, model structures, label spaces, and local data distributions.

The evaluation also investigates robustness across random seeds, communication efficiency, transfer to previously unseen clients, and scalability.

---

## PFTL Training Workflow

Each communication round follows the general workflow:

```text
              ┌─────────────────────┐
              │ Central Aggregator  │
              └──────────┬──────────┘
                         │
              Broadcast Shared Layer
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Client 1        Client 2       Client N
          │              │              │
          │         Local Training      │
          │              │              │
          └──────────────┼──────────────┘
                         │
               Upload Shared Layer
                         │
                         ▼
              ┌─────────────────────┐
              │   Weighted FedAvg   │
              └──────────┬──────────┘
                         │
                 Global Shared Layer
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       γ-blend         γ-blend        γ-blend
          │              │              │
       Validate        Validate       Validate
          │              │              │
      Accept/Keep     Accept/Keep    Accept/Keep
```

At each round:

1. The server broadcasts the current shared-layer parameters.
2. Each client initializes its local shared layer using the received parameters.
3. Clients train their models locally.
4. Only the updated shared-layer parameters are uploaded.
5. The server performs sample-size-weighted aggregation.
6. The aggregated shared parameters are returned to the clients.
7. Each client performs local–global γ-blending.
8. In the adaptive setting, the client evaluates the blended representation using local validation data.
9. The personalized representation is used to initialize the next local round.

---

## Model Architecture

Conceptually, client \(i\) follows:

```text
x_i
 │
 ▼
Private Encoder
 │
 ▼
Private Adapter: Dense(*)
 │
 ▼
Shared Layer: Dense(q)
 │
 ▼
Private Classifier: Dense(C_i)
 │
 ▼
Sigmoid / Softmax
 │
 ▼
Prediction
```

### Private Feature Encoder

The implementation uses private feature-extraction components based on:

```text
Conv1D
  ↓
MaxPool1D
  ↓
GlobalAveragePooling1D
```

These layers remain client-specific and can operate on heterogeneous local feature spaces.

### Private Adapter

A private `Dense(*)` adapter maps the client-specific representation to the input dimensionality required by the common shared interface.

The adapter itself is **not aggregated**.

### Shared Layer

The `shared_dense` layer is the only model component synchronized across clients.

The implementation uses:

```text
Phase One: q = 4
Phase Two: q = 8
```

### Private Classification Head

Each client maintains its own classifier.

For binary classification:

```text
Dense(1) + Sigmoid
```

For multi-class classification:

```text
Dense(C_i) + Softmax
```

Because the classifier remains private, clients can operate with different label spaces and attack taxonomies.

---

## Experimental Design

The framework is evaluated using six intrusion-detection datasets representing heterogeneous IoT and network environments:

* **CIC-IoT-2022**
* **CIC-BCCC-NRC-2024**
* **CIC-IoT-2023**
* **UNSW-NB15**
* **TON-IoT**
* **CIC-IDS-2017**

The evaluation is divided into two complementary phases.

---

## Phase One — Controlled Binary Setting

Phase One evaluates PFTL in a controlled binary-classification environment.

In this phase, participating clients use:

* compatible model architectures,
* aligned binary label spaces, and
* fixed γ-blending coefficients.

This controlled setting enables direct comparison with conventional federated and personalized federated learning baselines.

Different combinations of local and global blending coefficients are evaluated to study the trade-off between local specialization and collaborative learning.

---

## Phase Two — Heterogeneous Multi-Class Setting

Phase Two evaluates PFTL under substantially stronger heterogeneity.

Participating clients may differ in:

| Type of Heterogeneity | Supported |
| --------------------- | :-------: |
| Feature spaces        |     ✓     |
| Model architectures   |     ✓     |
| Label spaces          |     ✓     |
| Non-IID distributions |     ✓     |

In this setting, PFTL uses **adaptive γ-blending** together with the **validation-based safety mechanism**.

The adaptive mechanism is initialized with:

```text
γ_global = 0.50
γ_local  = 0.50
```

with:

```text
η       = 0.05
τ       = 0.05
γ_min   = 0.10
γ_max   = 0.90
ε       = 0.001
```

These parameters allow the degree of collaborative transfer to evolve according to each client's validation behavior.

---

## Baselines

The evaluation considers representative federated and personalized learning approaches, including:

* **Standalone Learning**
* **FedAvg**
* **FedPer**
* **FedRep**
* **FedClassAvg**
* **FedProto**

The experimental design distinguishes between methods that require stronger structural compatibility and approaches that can operate under different forms of heterogeneity.

---

## Communication Efficiency

A central design goal of PFTL is to avoid unnecessary full-model communication.

If the shared layer contains:

```text
W_s ∈ R^(p × q)
b_s ∈ R^q
```

only:

```text
(W_s, b_s)
```

are exchanged.

The private parameters associated with the encoder, adapter, and classifier remain local.

Measured serialized communication:

```text
Binary setting      ≈ 606 bytes/client/round
Multi-class setting ≈ 1,404 bytes/client/round
```

This communication cost is determined primarily by the dimensionality of the compact shared interface rather than the size of each client's complete private architecture.

---

## Scalability

PFTL is also evaluated under increasing federation sizes.

The scalability experiment increases the federation from **6 to 90 clients** while evaluating predictive performance under heterogeneous client distributions.

Reported Macro-F1 remains relatively stable:

```text
6 clients  → Macro-F1 ≈ 0.487
90 clients → Macro-F1 ≈ 0.464
```

These experiments examine whether the compact representation-sharing mechanism can maintain useful collaborative learning as the number of participating clients increases.

---

## Unseen-Client Transfer

The evaluation additionally investigates whether the learned shared representation can transfer useful knowledge to a client that was not part of the original federation.

This experiment evaluates the transferability of the learned compact shared representation under distribution shift and provides evidence that the shared layer captures information that can be useful beyond the original participating clients.

---

## Datasets

The datasets are **not redistributed in this repository**.

Please obtain the datasets from their original providers and configure the corresponding paths locally before running the experiments.

The study uses:

```text
CIC-IoT-2022
CIC-BCCC-NRC-2024
CIC-IoT-2023
UNSW-NB15
TON-IoT
CIC-IDS-2017
```

---

## Repository Structure

The final repository may be organized as follows:

```text
PFTL/
│
├── README.md
├── requirements.txt
│
├── datasets/
│   └── README.md
│
├── phase_one/
│   ├── clients/
│   ├── server/
│   ├── baselines/
│   └── experiments/
│
├── phase_two/
│   ├── clients/
│   ├── server/
│   ├── models/
│   └── experiments/
│
├── scalability/
│
├── unseen_client/
│
├── utils/
│
└── results/
```

> **Note:** This structure should be updated to match the final released implementation.

---

## Installation

Clone the repository:

```bash
git clone <REPOSITORY-URL>
cd PFTL
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

The exact execution commands depend on the final organization of the released code.

A typical client–server experiment follows:

### 1. Start the Aggregator

```bash
python server.py
```

### 2. Start the Clients

```bash
python client.py
```

### 3. Run the Required Experimental Configuration

Separate configurations can be provided for:

```text
Phase One
Phase Two
Scalability
Unseen-client transfer
Baseline experiments
```

> The commands in this section should be replaced with the exact filenames and arguments from the released implementation.

---

## Reproducibility

The experimental framework evaluates PFTL across several dimensions:

* predictive performance,
* personalization,
* feature-space heterogeneity,
* model heterogeneity,
* label-space heterogeneity,
* non-IID distributions,
* negative transfer,
* unseen-client transfer,
* communication efficiency,
* scalability, and
* robustness across multiple random seeds.

For exact experimental settings, preprocessing procedures, hyperparameters, and statistical analyses, please refer to the accompanying paper.

---

## Important Scope Note

PFTL is designed to address **heterogeneous and personalized federated knowledge transfer**.

The method does **not introduce a new privacy-preserving mechanism** such as differential privacy, secure aggregation, homomorphic encryption, or another cryptographic privacy protocol.

As in the standard federated-learning setting considered in this work, raw datasets remain local to the participating clients. The methodological contributions of PFTL concern **heterogeneity handling, personalized knowledge transfer, compact representation-level synchronization, validation-driven adaptation, communication efficiency, and robustness**.

---

## Citation

If you use PFTL or this implementation in your research, please cite the accompanying paper:

```bibtex
@article{alqahtani2026pftl,
  title   = {Personalized Federated Transfer Learning for Intrusion Detection
             across Networks with Heterogeneous Feature Spaces,
             Model Architectures, and Label Spaces},
  author  = {Alqahtani, Azizah and
             Aljoby, Walid and
             Ragab, Mohamed and
             Brik, Bouziane and
             Felamban, Muhamad and
             Helmy, Tarek},
  year    = {2026}
}
```

The journal, volume, pages, and DOI should be added once the final bibliographic information is available.

---

## Authors

**Azizah Alqahtani**
King Fahd University of Petroleum & Minerals (KFUPM), Saudi Arabia
Ministry of Education, Saudi Arabia

**Walid Aljoby**
King Fahd University of Petroleum & Minerals (KFUPM), Saudi Arabia

**Mohamed Ragab**
Technology Innovation Institute, Abu Dhabi, United Arab Emirates

**Bouziane Brik**
University of Sharjah, United Arab Emirates

**Muhamad Felamban**
King Fahd University of Petroleum & Minerals (KFUPM), Saudi Arabia

**Tarek Helmy**
King Fahd University of Petroleum & Minerals (KFUPM), Saudi Arabia

---

## License

Please refer to the repository license for the terms governing use of the released source code.

---

## Acknowledgment

This repository accompanies the research on **Personalized Federated Transfer Learning (PFTL)** for intrusion detection across clients with heterogeneous feature spaces, model architectures, label spaces, and data distributions.
