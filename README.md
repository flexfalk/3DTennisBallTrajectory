# TennisTrajectoryReconstruction

This is the official repository of the paper with the title [*SynthNet: Leveraging Synthetic Data for 3D Trajectory Estimation from Monocular Video*](https://dl.acm.org/doi/10.1145/3689061.3689073). 

Note that it is open access!


## Reference 

When using this code or parts of it, please cite the corresponding publication.

### ACM
Morten Holck Ertner, Sofus Schou Konglevoll, Magnus Ibh, and Stella Graßhof. 2024. SynthNet: Leveraging Synthetic Data for 3D Trajectory Estimation from Monocular Video. In Proceedings of the 7th ACM International Workshop on Multimedia Content Analysis in Sports (MMSports '24). Association for Computing Machinery, New York, NY, USA, 51–58. https://doi.org/10.1145/3689061.3689073

### Bibtex

@inproceedings{10.1145/3689061.3689073,
author = {Ertner, Morten Holck and Konglevoll, Sofus Schou and Ibh, Magnus and Gra{\ss}hof, Stella},
title = {SynthNet: Leveraging Synthetic Data for 3D Trajectory Estimation from Monocular Video},
year = {2024},
isbn = {9798400711985},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3689061.3689073},
doi = {10.1145/3689061.3689073},
abstract = {Reconstructing 3D trajectories from video is often cumbersome and expensive, relying on complex or multi-camera setups. This paper proposes SynthNet, an end-to-end pipeline for monocular reconstruction of 3D tennis ball trajectories. The pipeline consists of two parts: Hit and bounce detection and 3D trajectory reconstruction. The hit and bounce detection is performed by a GRU-based model, which segments the videos into individual shots. Next, a fully connected neural network reconstructs the 3D trajectory through a novel physics-based training approach relying on purely synthetic training data. Instability in the training loop caused by relying on Euler-time integration and camera projections is circumvented by our synthetic approach, which directly calculates loss from estimated initial conditions, improving stability and performance. In experiments, SynthNet is compared to an existing reconstruction baseline on a number of conventional and customized metrics defined to validate our synthetic approach. SynthNet outperforms the baseline based on our own proposed metrics and in a qualitative inspection of the reconstructed 3D trajectories.},
booktitle = {Proceedings of the 7th ACM International Workshop on Multimedia Content Analysis in Sports},
pages = {51–58},
numpages = {8},
keywords = {3d reconstruction, ball tracking, computer vision, differential equations, machine learning in sports, neural network, synthetic data},
location = {Melbourne VIC, Australia},
series = {MMSports '24}
}
