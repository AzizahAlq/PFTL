# Personalized Federated Transfer Learning (PFTL)

Official implementation of the paper:

"Personalized Federated Transfer Learning for Intrusion Detection across Networks with Heterogeneous Feature Spaces, Model Architectures, and Label Spaces"

Authors: Azizah Alqahtani, Walid Aljoby, Mohamed Ragab, Bouziane Brik, Muhamad Felamban, and Tarek Helmy

---

## Overview

This repository provides the implementation and experimental framework for Personalized Federated Transfer Learning (PFTL), a federated learning approach designed for intrusion detection across heterogeneous network environments.

Traditional Federated Learning (FL) generally relies on structural compatibility among participating clients. In realistic intrusion-detection environments, however, clients may use different traffic features, preprocessing pipelines, local model architectures, attack classes, and data distributions.

PFTL addresses these challenges by supporting clients with heterogeneous:

- feature spaces,
- model architectures,
- label spaces, and
- non-IID data distributions.

The main idea is to keep most of each client's model private and exchange only the parameters of a compact shared layer.

Each client maintains its own feature encoder, private adapter, and classifier head. Only the shared-layer parameters are exchanged with the server and aggregated during federated training.

This provides a lightweight interface for knowledge transfer without requiring clients to have identical input dimensions, private architectures, or output label spaces.

---

## PFTL Architecture

Each client follows the general architecture:

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

Only the shared-layer parameters are communicated between each client and the central aggregator.

The following components remain local:

- raw client data,
- private feature encoder,
- private adapter,
- private classifier head.

Therefore, clients only need to agree on the dimensionality of the shared layer.

---

## Main Features

### Heterogeneous Federated Learning

PFTL supports simultaneous heterogeneity in:

- feature spaces,
- private model architectures,
- label spaces,
- attack taxonomies, and
- local data distributions.

### Shared-Layer Parameter Exchange

Instead of communicating the complete local model, PFTL exchanges only the parameters of a compact shared layer.

The private encoder, adapter, and classifier remain client-specific.

### Personalized Local–Global Blending

After receiving the aggregated shared-layer parameters, each client combines its locally learned shared-layer parameters with the global candidate.

For client \(i\):

```text
W_mixed = γ_local W_local + γ_global W_global
```

where:

```text
γ_local + γ_global = 1
```

The blending coefficients control the balance between local specialization and collaborative knowledge.

### Adaptive Personalization

Phase Two also evaluates adaptive γ-blending.

The global blending coefficient is adjusted according to the client's local validation behavior.

If the global candidate provides useful knowledge, its contribution can increase. If it is less beneficial, the client can shift toward its locally learned shared-layer parameters.

### Validation-Based Safety Gate

Each client evaluates the blended candidate using its own local validation data.

The candidate is accepted only when:

```text
F1_mixed >= F1_local + ε
```

Otherwise, the client keeps its locally updated shared-layer parameters.

This mechanism is intended to reduce harmful transfer under heterogeneous client conditions.

### Communication Efficiency

PFTL communicates only the shared-layer parameters rather than the complete client model.

Measured serialized communication in the evaluated configurations is approximately:

| Setting | Shared Dimension | Communication per Client per Round |
|---|---:|---:|
| Phase One / Binary | q = 4 | ~606 bytes |
| Phase Two / Multi-class | q = 8 | ~1,404 bytes |

---

## Training Workflow

Each communication round follows the general process:

```text
              ┌─────────────────────┐
              │ Central Aggregator  │
              └──────────┬──────────┘
                         │
                Broadcast Shared
                Layer Parameters
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Client 1        Client 2       Client N
          │              │              │
          └────── Local Training ───────┘
                         │
                Upload Shared-Layer
                    Parameters
                         │
                         ▼
              ┌─────────────────────┐
              │   Weighted FedAvg   │
              └──────────┬──────────┘
                         │
                 Global Shared-Layer
                     Parameters
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
2. Each client initializes or updates its shared layer using the received parameters.
3. Each client performs local training.
4. Only the updated shared-layer parameters are uploaded.
5. The server performs sample-size-weighted aggregation.
6. The aggregated shared-layer parameters are returned to the clients.
7. Each client performs local-global γ-blending.
8. The candidate is evaluated using local validation data.
9. The accepted personalized shared-layer parameters are used for the next round.

---

## Experimental Design

The evaluation is divided into two main phases.

### Phase One — Controlled Binary Setting

Phase One evaluates PFTL in a controlled binary-classification environment.

This phase provides comparisons with conventional federated and personalized federated-learning baselines.

The Phase One implementation and baselines are organized under:

```text
phase_one/
├── pftl/
│   └── core/
├── baselines/
│   ├── fedavg/
│   ├── fedclassavg/
│   ├── fedper/
│   └── fedrep/
└── experiments/
```

Different local/global blending configurations can be evaluated to study the effect of personalization.

### Phase Two — Heterogeneous Multi-Class Setting

Phase Two evaluates PFTL under stronger heterogeneity.

Clients may differ in:

| Type of Heterogeneity | Supported |
|---|:---:|
| Feature spaces | ✓ |
| Model architectures | ✓ |
| Label spaces | ✓ |
| Non-IID distributions | ✓ |

The current adaptive configuration uses:

```text
γ_global_init = 0.50
γ_local_init  = 0.50

η       = 0.08
τ       = 0.02
γ_min   = 0.10
γ_max   = 0.90
ε       = 0.001
```

Phase Two also contains experiments comparing static and adaptive personalization and experiments involving FedProto.

---

## Baselines

The repository contains implementations and experiments involving the following methods:

- Standalone learning
- FedAvg
- FedPer
- FedRep
- FedClassAvg
- FedProto
- PFTL

The available baseline depends on the experimental phase and the type of heterogeneity being evaluated.

---

## Datasets

The repository is organized into separate dataset directories for Phase One and Phase Two:

```text
datasets/
├── phase_one_datasets/
└── phase_two_datasets/
```

### Phase One Datasets

The Phase One directory contains the prepared datasets used for the controlled binary experiments.

### Phase Two Datasets

The Phase Two directory contains the prepared datasets used for the heterogeneous multi-class experiments.

The current Phase Two dataset collection includes prepared data corresponding to:

- CIC-ToN-IoT
- CIC-IoT-2023
- UNSW-NB15
- CIC-IDS-2017
- CIC-BCCC-NRC-2024
- CIC-IoT-IDaD-2024

The Phase Two dataset directory also contains the generated ToN-IoT virtual-client data used by the corresponding experiment:

```text
datasets/phase_two_datasets/ton_iot_10_virtual_clients/
```

Dataset preprocessing and client-partitioning scripts are provided where applicable.

Users of the repository should also refer to the original dataset providers for dataset descriptions, licensing conditions, and citation requirements.

---

## Repository Structure

The repository is currently organized as follows:

```text
PFTL/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── datasets/
│   ├── phase_one_datasets/
│   └── phase_two_datasets/
│       └── ton_iot_10_virtual_clients/
│
├── phase_one/
│   ├── pftl/
│   │   └── core/
│   ├── baselines/
│   │   ├── fedavg/
│   │   ├── fedclassavg/
│   │   ├── fedper/
│   │   └── fedrep/
│   └── experiments/
│
├── phase_two/
│   ├── pftl/
│   │   └── core/
│   ├── baselines/
│   │   └── fedproto/
│   │       ├── experiment_1/
│   │       └── experiment_2/
│   ├── experiments/
│   │   └── static_vs_adaptive/
│   ├── FedProto_experiment1/
│   └── FedProtoVsPFTL_experiment2/
│
├── unseen_client/
│   ├── binary/
│   └── multiclass/
│
├── scalability/
│
├── statistical_analysis/
│
├── results/
│   ├── phase_one/
│   ├── phase_two/
│   ├── scalability/
│   └── unseen_client/
│
└── utils/
```

---

## Requirements

The implementation uses Python and the following main packages:

```text
numpy
pandas
tensorflow
scikit-learn
grpcio
protobuf
matplotlib
seaborn
```

Python standard-library packages such as `os`, `csv`, `pickle`, `threading`, `random`, `time`, `json`, `pathlib`, and `datetime` are also used but do not require separate installation.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AzizahAlq/PFTL.git
cd PFTL
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running PFTL

PFTL uses a client-server architecture implemented with gRPC.

The general execution procedure is:

1. Start the corresponding aggregator/server.
2. Start the participating clients.
3. Allow the clients to perform local training.
4. Exchange only the shared-layer parameters.
5. Aggregate the received shared-layer parameters.
6. Perform local-global personalization at each client.
7. Evaluate the personalized candidate using the validation gate.
8. Continue until the configured number of communication rounds is completed.

Because the repository contains multiple experimental configurations, use the scripts in the corresponding Phase One, Phase Two, baseline, unseen-client, or scalability directory.

---

## Static vs Adaptive Experiment

The Phase Two static-versus-adaptive experiments are located under:

```text
phase_two/experiments/static_vs_adaptive/
```

These experiments investigate the effect of using fixed γ coefficients compared with validation-driven adaptive γ values.

---

## FedProto Experiments

Phase Two also contains experiments involving FedProto.

These experiments are used to evaluate PFTL relative to prototype-based federated knowledge sharing under heterogeneous conditions.

The corresponding code is located in the Phase Two experiment and baseline directories.

---

## Unseen-Client Evaluation

The repository contains separate unseen-client experiments:

```text
unseen_client/
├── binary/
└── multiclass/
```

These experiments investigate whether shared-layer parameters learned during federation can provide useful transferable knowledge to clients that did not participate in the original federated training.

The evaluation includes local standalone training and transfer-based configurations.

---

## Scalability

Scalability experiments evaluate PFTL as the number of participating clients increases.

The scalability evaluation investigates whether the compact shared-layer parameter exchange remains effective as the federation grows.

Related code and results are organized under:

```text
scalability/
results/scalability/
```

---

## Statistical Analysis

Statistical analyses associated with the experiments are provided under:

```text
statistical_analysis/
```

These analyses are used to evaluate the consistency and statistical significance of the experimental comparisons across clients and repeated runs.

---

## Reproducibility

The experimental framework evaluates PFTL across several dimensions, including:

- predictive performance,
- personalization,
- feature-space heterogeneity,
- model heterogeneity,
- label-space heterogeneity,
- non-IID distributions,
- negative transfer,
- unseen-client transfer,
- communication efficiency,
- scalability, and
- robustness across multiple random seeds.

Random seeds, experimental parameters, dataset preprocessing, and client configurations should be kept consistent when reproducing the reported experiments.

For the complete experimental methodology and statistical analysis, please refer to the accompanying paper.

---

## Scope

PFTL is designed to address heterogeneous and personalized federated knowledge transfer.

The current framework does not introduce a new privacy-preserving mechanism such as differential privacy, secure aggregation, or homomorphic encryption.

Raw data are not exchanged among participating clients during federated training. The primary methodological contributions concern heterogeneous collaboration, personalized knowledge transfer, shared-layer parameter exchange, validation-driven adaptation, communication efficiency, and robustness.

---

## Citation

If you use PFTL or this implementation in your research, please cite the accompanying paper:

```bibtex
@article{alqahtani2026pftl,
  title  = {Personalized Federated Transfer Learning for Intrusion Detection
            across Networks with Heterogeneous Feature Spaces,
            Model Architectures, and Label Spaces},
  author = {Alqahtani, Azizah and
            Aljoby, Walid and
            Ragab, Mohamed and
            Brik, Bouziane and
            Felamban, Muhamad and
            Helmy, Tarek},
  year   = {2026}
}
```

The final journal, volume, pages, and DOI information can be added when available.

---

## Authors

Azizah Alqahtani  
King Fahd University of Petroleum & Minerals (KFUPM), Saudi Arabia  
Ministry of Education, Saudi Arabia

Walid Aljoby  
King Fahd University of Petroleum & Minerals (KFUPM), Saudi Arabia

Mohamed Ragab  
Technology Innovation Institute, Abu Dhabi, United Arab Emirates

Bouziane Brik  
University of Sharjah, United Arab Emirates

Muhamad Felamban  
King Fahd University of Petroleum & Minerals (KFUPM), Saudi Arabia

Tarek Helmy  
King Fahd University of Petroleum & Minerals (KFUPM), Saudi Arabia

---

## Acknowledgment

This repository accompanies the research on Personalized Federated Transfer Learning (PFTL) for intrusion detection across heterogeneous network environments.
