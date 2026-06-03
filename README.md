# FUSE: Adaptive Frequency-Temporal Unified Semantic Attacks for Time Series Forecasting

This project is developed based on [BasicTS](https://github.com/GestaltCogTeam/BasicTS). It implements FUSE, an adaptive frequency-temporal unified semantic attack method specifically designed for time series forecasting models.

## 🚀 Quick Start

For detailed instructions on environment configuration, dataset preparation, and base model training, please refer to:
👉 **[Getting Started](./tutorial/getting_started.md)**

## 🛠 Adversarial Attack Experiments

The project integrates a variety of time series adversarial attack algorithms, including **FGSM, BIM, MI-FGSM, NI-FGSM, Fre-C&W, TCA, MAPGD, AAIM, and FUSE**. You can conduct experiments directly by running the following scripts:

### 1. White-box Attack
To perform white-box attacks on models where you have full access to parameters and gradients, please run:
```bash
python run_whitebox_suite.py
```

### 2. Black-box Attack
To perform black-box attacks on models where you only have access to inputs and outputs (limited interaction), please run:
```bash
python run_blackbox_suite.py
```

## 🔗 Acknowledgement

The core code framework of this project relies on [BasicTS](https://github.com/GestaltCogTeam/BasicTS). We would like to thank the development team for their outstanding open-source contribution.
