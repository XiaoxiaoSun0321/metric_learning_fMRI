# Functional Neuroimaging with Deep Metric Learning

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

We propose a metric learning framework to extract meaningful latent structures from high-dimensional fMRI data. This method learns the latent embeddings that reduce the intra-group variability while maximizing the inter-group variability. More details can be found here (new manuscript is in preparation): https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9871492

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)
- [Contact](#contact)

## Installation

All presented codes are implemented in Python 3.7 (version>=3.7 might run into package compatibility issue). 

needed packages:
pytorch=1.12.1
nilearn=0.10.1
umap-learn==0.5.3
pynndescent

## Usage

Codes are provided in Jupyter notebook. Example data (from ABIDE-150) is also provided. 

## Features

- Feature 1: step-by-step code to learn embeddings by using metric learning
- Feature 2: comparable result from embedding learned by using principle component analysis (PCA)
- Feature 3: Umap visualization of the learned latent space (in lower dimension) learned from both methods

## Contributing

We welcome contributions from the community! If you would like to contribute to the project, please let us know.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Refer to the [LICENSE](LICENSE) file for more information.

## Credits

Please cite our previous paper if used (new manuscript is in preparation): https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9871492


Example data is ABIDE-150 from _Multisite functional connectivity mri classification of autism: Abide results_. Please cite the paper if used: https://www.frontiersin.org/articles/10.3389/fnhum.2013.00599/full


This work was supported by a Vannevar Bush Faculty Fellowship from the US Department of Defense (N00014-20-1-2027) and a Center of Excellence grant from the Air Force Office of Scientific Research (FA9550-22-1-0337). 

## Contact

For any inquiries or questions, you can reach us at [arunesh.mitl@gmail.com] and [xiaoxiao.sun@columbia.edu]. 
