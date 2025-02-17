# Cluster-guided contrastive learning with masked antoencoder for spatial domain identification based on spatial transcriptomics
The complete code will be made available after the article published.
## Overview:

STMCCL a novel self-supervised learning framework that jointly trains a masked autoencoder and cluster-guided contrastive learning.STMCCL has been applied to seven spatial transcriptomics datasets across platforms like 10X Visium, Stereo-seq, and Slide-seqV2, proving its capability to deliver enhanced representations for a range of downstream analyses, such as clustering, visualization, trajectory inference, and differential gene analysis.

## Requirements:
 
STMCCL is implemented in the pytorch framework. Please run STMCCL on CUDA. The following packages are required to be able to run everything in this repository (included are the versions we used):

```bash
python==3.10.0
torch==2.1.0
cuda==11.8
numpy==1.25.2
scanpy==1.10.4
pandas==2.2.3
scipy==1.14.1
scikit-learn==1.5.2
anndata==0.11.0
R==4.3.3
ryp2==3.5.12
tqdm==4.67.0
matplotlib==3.9.2
seaborn==0.13.2
```
