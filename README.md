# IMDiffusion

This repository is the implementation of IMDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection. We propose the IMDiffusion framework for unsupervised anomaly detection and evaluate its performance on six open-source datasets.

# Results
The main results are presented in the following table. Our method outperforms the previous unsupervised anomaly detection methods in the majority of metrics.

![Image Description](result.png)

# Train and inference


To reproduce the results mentioned in our paper, first, make sure you have torch and pyyaml installed in your environment. Then, use the following command to train:
```shell
python exe_machine.py --device cuda:0
```

The Dataset folder contains the necessary files for training, including train, test, and label files. In this repository, we provide file data for SMD.


After completing the training, you can use the

```shell
python evaluate_machine_window_middle.py --device cuda:0
```

command to perform inference.