# TransCNNLoc: End-to-End Pixel-level Learning for 2D-to-3D Pose Estimation in Dynamic Indoor Scenes

We propose the TransCNNLoc framework, which consists of an encoding-decoding network designed to **learn more robust image features** for camera pose estimation. 

- Submitted to ISPRS Journal of Photogrammetry and Remote Sensing
- Authors: Shengjun Tang<sup>*</sup>, **Yusong Li<sup>\*</sup>**, Jiawei Wan, You Li, Baoding Zhou, Renzhong Guo, Weixi Wang, Yuhong Feng

<p align="center">
  <img src="framework.png" width="600" height="400">
</p>

## Results
The experimental results demonstrate that the proposed TransCNNLoc framework exhibits superior adaptability to dynamic scenes and lighting changes, achieving the
best localization accuracy in all 7scenes datasets, with sub-10cm level accuracy.
<p align="center">
  <img src="table.jpg" >
</p>

## Installation
Installing the package locally also installs the minimal dependencies listed in ```requirements.txt```:
```
git clone https://github.com/Geelooo/TransCNNloc.git
cd TransCNNloc/
pip install -r requirements.txt
```


## Data Preparation
\begin{table}[htps]
\centering
\resizebox{0.95\textwidth}{!}{
\begin{tabular}{cccccccc}\hline
\textbf{}                                            & Chess                                 & Fire                                  & Heads                                 & Office                               & Pumpkin                               & Redkitchen                            & Stairs                                \\ \hline
Spatial Extent(m)                                    & 3×2×1m                                & 2.5×1×1m                              & 2×0.5×1m                              & 2.5×2×1.5m                           & 2.5×2×1m                              & 4×3×1.5m                              & 2.5×2×1.5m                            \\
\# Frames: Train,Test                                & 4000,2000                             & 2000,2000                             & 1000,1000                             & 6000,4000                            & 4000,2000                             & 9000,5000                             & 4000,2000                             \\ \hline
PoseNet\citep{kendall2015posenet}    & 0.32/4.06                             & 0.47/7.33                             & 0.29/6                                & 0.48/3.84                            & 0.47/4.21                             & 0.59/4.32                             & 0.47/6.93                             \\
PoseNet2\citep{kendall2017geometric} & 0.13/4.48                             & 0.27/11.3                             & 0.17/13                               & 0.19/5.55                            & 0.26/4.75                             & 0.23/5.35                             & 0.35/12.4                             \\
NNnet\citep{laskar2017camera}        & 0.13/6.46                             & 0.26/12.72                            & 0.14/12.34                            & 0.21/7.35                            & 0.24/6.35                             & 0.24/8.03                             & 0.27/11.82                            \\
RelocNet\citep{balntas2018relocnet}  & 0.12/4.14                             & 0.26/10.4                             & 0.14/10.5                             & 0.18/5.32                            & 0.26/4.17                             & 0.23/5.08                             & 0.28/7.53                             \\
MapNet\citep{brahmbhatt2018geometry} & 0.08/3.25                             & 0.27/11.69                            & 0.18/13.25                            & 0.17/5.15                            & 0.22/4.02                             & 0.23/4.93                             & 0.3/12.08                             \\
Pixloc\_cmu\citep{sarlin2021back}    & 0.16/6.068                            & 0.14/6.188                            & 0.154/9.71                            & 0.135/3.805                          & 0.16/4.154                            & 0.167/5.216                           & 0.275/7.499                           \\
\hl{GTCaR}\citep{li2022gtcar}                                                 & 0.09/1.94                             & 0.27/8.45                             & 0.12/9.34                             & 0.12/2.41                            & 0.15/2.13                             & 0.26/2.73                             & 0.26/8.92                             \\
\hl{MMLNet}\citep{wang2023deep}                                                & 0.08/3.15                             & 0.24/9.25                             & 0.16/11.1                             & 0.16/5.05                            & 0.18/4.02                             & 0.22/4.85                             & \textbf{0.23/7.49}                             \\
TransCNNLoc(Ours)                                          & \textbf{0.056/1.826} & \textbf{0.119/4.719} & \textbf{0.098/6.107} & \textbf{0.077/2.31} & \textbf{0.115/2.935} & \textbf{0.098/2.894} & 0.265/8.177 \\ \hline
\end{tabular}}
\caption{Localization accuracy compared to other end-to-end methods, represented as median error (m/°).}
\label{tab:compare}
\end{table}

To specify the file directories, we use ```settings.py``` and ```run_scripts.py``` files. The dependencies from [third-party libraries](https://drive.google.com/file/d/1pN3UVUmFwVBbtjbwc4bbUJ2hMVSH15ku/view?usp=sharing) need to be downloaded in advance. We utilize both publicly available datasets, such as 7Scenes, and our [own collected data](https://drive.google.com/file/d/1HrsrM5lpSFMHiy1KnnGmgiGAGnl3XxOH/view?usp=sharing). By doing so, we can evaluate and compare the performance across different datasets.

```
python -m run_scripts --scene=jiawei_cheku
```
To run and evaluate position accuracy.
- ```--scene``` parameter can be used to specify different datasets. 

The dataset should be organized according to the specified format mentioned above. Adjust the paths in the configuration file accordingly and evaluate different datasets as needed.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.


## Acknowledgements

Part of the code implementation was adapted from [Pixloc's repository](https://github.com/cvg/pixloc).

<!-- Please consider citing our work if you use any of the ideas presented the paper or code from this repo:

```
@misc{du2023asymformer,
      title={AsymFormer: Asymmetrical Cross-Modal Representation Learning for Mobile Platform Real-Time RGB-D Semantic Segmentation}, 
      author={Siqi Du and Weixi Wang and Renzhong Guo and Shengjun Tang},
      year={2023},
      eprint={2309.14065},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->