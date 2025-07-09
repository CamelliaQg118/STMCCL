# STMCCL(DOI:10.1109/JBHI.2025.3586483)
An official source code for the paper "Cluster-guided contrastive learning with masked autoencoder for spatial domain identification based on spatial transcriptomics," accepted by IEEE Journal of Biomedical and Health Informatics 2025 (https://ieeexplore.ieee.org/document/11072285). Any communications or issues are welcome. Please contact qigao118@163.com. If you find this repository useful to your research or work, it is really appreciated to star this repository. ❤️
## Overview:
![Gao1](https://github.com/user-attachments/assets/3e2542b7-79b5-40ed-8d02-369e1956b8bf)

STMCCL, a novel self-supervised learning framework that jointly trains a masked autoencoder and cluster-guided contrastive learning. STMCCL has been applied to seven spatial transcriptomics datasets across platforms like 10X Visium, Stereo-seq, and Slide-seqV2, proving its capability to deliver enhanced representations for a range of downstream analyses, such as clustering, visualization, trajectory inference, and differential gene analysis.

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
## Download dataset:
Go to https://zenodo.org/records/14911869, download all datasets, and put them into your local data directory.

## Citation:
If you find this repository helpful, please cite our paper.

