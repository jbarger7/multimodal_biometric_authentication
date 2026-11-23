
# Evaluating Multimodal Biometric Authentication Through Score-Level Fusion

Honors Thesis, Murray State University

## Project Overview

This project investigates whether score-level fusion of facial recognition and keystroke dynamics improves biometric verification accuracy.

The experiments compare face-only, keystroke-only, and fused verification scores using two public biometric datasets.

## Datasets

**CMU Keystroke Dynamics Benchmark Dataset**
https://www.cs.cmu.edu/~keystroke/

The original CSV file is `DSL-StrongPasswordData.csv`.

**ORL/AT&T Olivetti Faces Dataset**

Loaded through `sklearn.datasets.fetch_olivetti_faces(shuffle=False)`.

The datasets contain different participants, so the experiments use artificially paired (chimeric) identities.

## Source Code

- `thesis_experiment.py`: Primary experiment using 50 PCA components.
- `thesis_experiment_pca110.py`: Sensitivity experiment using 110 PCA components.
- `thesis_experiment_pairing.py`: Alternative synthetic identity pairings using random seeds 123, 456, and 789.

## Software Requirements

Install the Python dependencies using:

    python -m pip install -r requirements.txt

The preserved project environment uses Python 3.13.7.

## Dataset Setup

Download the CMU keystroke dataset and place the CSV file at:

    data/DSL-StrongPasswordData.csv

The Olivetti Faces dataset is loaded using scikit-learn.

## Experimental Results

Primary held-out evaluation:

| Method | EER |
|---|---|
| Face only | 11.65% |
| Keystroke only | 19.17% |
| Score-level fusion | 8.33% |

Additional experiments evaluate the effects of PCA dimensionality and randomized identity pairings.

## Reproducibility

The repository includes experiment scripts, dependency versions, and generated results.

The reported results were obtained from a controlled chimeric benchmark rather than a real-world continuous authentication system.

## Limitations

The facial and keystroke observations come from different people. Therefore, the findings do not establish performance on genuinely matched multimodal participants or a deployed zero-trust system.
