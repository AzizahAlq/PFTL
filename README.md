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

### Personalized Local-Global Blending

After receiving the aggregated shared-layer parameters, each client combines its locally learned shared-layer parameters with the global candidate.

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

The global blending coefficient is adjusted according to the client's local validation behavior. If collaborative knowledge improves validation performance, its contribution can increase. Otherwise, the client can shift toward its locally learned shared-layer parameters.

The exact adaptive parameters used in each experiment are defined in the corresponding scripts.

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

The phase includes PFTL and the following baseline implementations:

- Standalone
- FedAvg
- FedClassAvg
- FedPer
- FedRep

The implementation is organized under:

```text
phase_one/
├── pftl/
│   └── core/
└── baselines/
    ├── standalone/
    ├── fedavg/
    ├── fedclassavg/
    ├── fedper/
    └── fedrep/
```

### Phase Two — Heterogeneous Multi-Class Setting

Phase Two evaluates PFTL under stronger heterogeneity.

Clients may differ in:

| Type of Heterogeneity | Supported |
|---|:---:|
| Feature spaces | ✓ |
| Model architectures | ✓ |
| Label spaces | ✓ |
| Non-IID distributions | ✓ |

Phase Two contains the main multi-class PFTL implementation together with additional experiments for adaptive personalization and FedProto comparison.

---

## Baselines

The repository contains implementations or experimental comparisons involving:

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

Datasets are organized into separate directories for Phase One and Phase Two:

```text
datasets/
├── phase_one_datasets/
└── phase_two_datasets/
```

### Phase One Datasets

The Phase One directory contains the prepared datasets used for the controlled binary experiments.

### Phase Two Datasets

The Phase Two directory contains the prepared datasets used for the heterogeneous multi-class experiments.

The Phase Two collection includes prepared data corresponding to:

- CIC-ToN-IoT
- CIC-IoT-2023
- UNSW-NB15
- CIC-IDS-2017
- CIC-BCCC-NRC-2024
- CIC-IoT-IDaD-2024

The directory also contains:

```text
create_10_stratified_clients.py
ton_iot_10_virtual_clients/
```

The `ton_iot_10_virtual_clients/` directory contains the generated virtual-client datasets used in the corresponding experiment.

Users should also refer to the original dataset providers for dataset descriptions, licensing conditions, and citation requirements.

---

## Repository Structure

The main GitHub repository is organized as follows:

```text
PFTL/
│
├── README.md
├── requirements.txt
├── .gitattributes
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
│   └── baselines/
│       ├── standalone/
│       ├── fedavg/
│       ├── fedclassavg/
│       ├── fedper/
│       └── fedrep/
│
├── phase_two/
│   ├── pftl/
│   │   └── core/
│   ├── experiments/
│   │   └── static_vs_adaptive/
│   ├── FedProto_experiment1/
│   │   └── FedProto_Global_Mapping/
│   └── FedProtoVsPFTL_experiment2/
│
├── scalability/
│
├── statistical_analysis/
│
└── unseen_client/
    ├── binary/
    └── multiclass/
```

---

## Requirements

The main Python packages used by the implementation are:

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

They can be installed using:

```bash
pip install -r requirements.txt
```

Python standard-library packages such as `os`, `csv`, `pickle`, `threading`, `random`, `time`, `json`, `pathlib`, and `datetime` do not require separate installation.

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

## Important: Update Dataset and Output Paths

Before running any experiment, the dataset paths in the corresponding Python scripts must be changed to match the dataset locations on the user's system.

The original experimental scripts may contain local or server-specific paths used during development, for example:

```python
DATA_PATH = "/nfs/.../dataset.csv"
```

These paths will not automatically work on another system.

They should be replaced with the appropriate path on the user's machine, for example:

```python
DATA_PATH = "/path/to/PFTL/datasets/phase_two_datasets/D1_CIC-ToN-IoT_new.csv"
```

The dataset paths must be checked and updated for all experiments, including:

- Phase One PFTL
- Phase One standalone experiments
- Phase One federated baselines
- Phase Two PFTL
- static vs. adaptive experiments
- FedProto experiments
- unseen-client experiments
- scalability experiments

Some scripts may also contain environment-specific paths for:

- output files,
- saved models,
- checkpoints,
- logs,
- mapping files,
- intermediate files, and
- transferred parameters.

These paths should also be changed according to the user's local environment before execution.

Users do not need to reproduce the original directory paths used during development.

---

## Running PFTL

Before starting an experiment, verify that all dataset, output, model, and log paths in the corresponding scripts have been updated for the local environment.

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

Because the repository contains multiple experimental configurations, use the scripts in the corresponding Phase One, Phase Two, unseen-client, or scalability directory.

---

## Static vs. Adaptive Experiment

The Phase Two static-versus-adaptive experiments are organized under:

```text
phase_two/experiments/static_vs_adaptive/
```

These experiments investigate the effect of using fixed γ coefficients compared with validation-driven adaptive γ values.

The exact hyperparameter values used for each experiment are defined in the corresponding scripts.

---

## FedProto Experiments

Phase Two contains two FedProto-related experimental configurations.

### Experiment 1

```text
phase_two/FedProto_experiment1/
```

This directory contains the FedProto implementation, corresponding PFTL configurations, and semantic label-mapping utilities.

The generated global semantic mapping files are organized under:

```text
phase_two/FedProto_experiment1/FedProto_Global_Mapping/
```

### Experiment 2 — FedProto vs. PFTL

```text
phase_two/FedProtoVsPFTL_experiment2/
```

This experiment contains FedProto, PFTL, and standalone client configurations used for experimental comparison.

It also includes scripts for collecting results across multiple random seeds.

---

## Unseen-Client Evaluation

Unseen-client experiments are separated into binary and multi-class configurations:

```text
unseen_client/
├── binary/
└── multiclass/
```

These experiments investigate whether shared-layer parameters learned during federation can provide useful transferable knowledge to clients that did not participate in the original federated training.

The evaluation includes standalone/local learning and transfer-based configurations.

---

## Scalability

Scalability experiments evaluate PFTL as the number of participating clients increases.

The corresponding code is organized under:

```text
scalability/
```

The generated ToN-IoT virtual-client datasets used by the corresponding scalability experiments are available under:

```text
datasets/phase_two_datasets/ton_iot_10_virtual_clients/
```

---

## Statistical Analysis

Statistical analysis files associated with the experimental evaluation are organized under:

```text
statistical_analysis/
```

These analyses are used to evaluate the consistency and statistical significance of experimental comparisons across clients and repeated runs.

---

## Reproducibility

The experimental framework evaluates PFTL across several dimensions, including:

- predictive performance,
- personalization,
- feature-space heterogeneity,
- model heterogeneity,
- label-space heterogeneity,
- non-IID distributions,
- unseen-client transfer,
- communication efficiency,
- scalability, and
- robustness across multiple random seeds.

When reproducing an experiment:

1. Use the corresponding prepared dataset.
2. Update the dataset and output paths.
3. Use the parameters defined in the corresponding experiment script.
4. Keep the random seed fixed when reproducing a specific run.
5. Start the appropriate aggregator before the federated clients.
6. Keep the client and server gRPC configurations consistent.
7. Use the same preprocessing and client-partitioning configuration.

For the complete experimental methodology and statistical analysis, please refer to the accompanying paper.

---

## Scope

PFTL is designed to address heterogeneous and personalized federated knowledge transfer.

The current framework does not introduce a new privacy-preserving mechanism such as differential privacy, secure aggregation, or homomorphic encryption.

Raw data are not exchanged among participating clients during federated training.

The primary methodological contributions concern heterogeneous collaboration, personalized knowledge transfer, shared-layer parameter exchange, validation-driven adaptation, communication efficiency, and robustness.

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
