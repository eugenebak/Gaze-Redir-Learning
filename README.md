# [WACV 2023] Fine Gaze Redirection Learning with Gaze Hardness-aware Transformation


Official PyTorch implementation of "**Fine Gaze Redirection Learning with Gaze Hardness-aware Transformation**"

<!-- ## 1. Requirements
### Environments

### Datasets

## 2. Training & Evaluation -->

## Abstract
Gaze redirection is a task to adjust the gaze of a given face or eye image toward the desired direction and aims to learn the gaze direction of a face image through a neural network-based generator. Considering that the prior arts have learned coarse gaze directions, learning fine gaze directions is very challenging. In addition, explicit discriminative learning of high-dimensional gaze features has not been reported yet. This paper presents solutions to overcome the above limitations. First, we propose the feature-level transformation which provides gaze features corresponding to various gaze directions in the latent feature space. Second, we propose a novel loss function for discriminative learning of gaze features. Specifically, features with insignificant or irrelevant effects on gaze (e.g., head pose and appearance) are set as negative pairs, and important gaze features are set as positive pairs, and then pair-wise similarity learning is performed. As a result, the proposed method showed a redirection error of only 2° for the GazeCapture dataset. This is a 10% better performance than a state-of-the-art method, i.e., STED. Additionally, the rationale for why latent features of various attributes should be discriminated is presented through activation visualization.
