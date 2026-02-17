# Zebrafish EM Analysis Toolkit

## Overview
This repository contains a comprehensive suite of algorithms and tools developed for the processing, reconstruction, and analysis of large-scale electron microscopy (EM) data, with a specific focus on zebrafish connectomics. 

The toolkit covers the entire pipeline from raw image quality assessment and denoising to neuron segmentation, organelle detection, and downstream data clustering.

## Modules

The codebase is organized into the following key components:

### 1. Image Enhancement & Denoising
* **`Blind2Sound` (EM Denoising v2.0)**
    * Our advanced, self-supervised denoising algorithm designed specifically for electron microscopy images. This version improves upon previous iterations by effectively suppressing noise without introducing residual artifacts, preserving high-frequency structural details.
* **`Blind2Unblind` (EM Denoising v1.0)**
    * The baseline version of our EM image denoising framework. It provides the foundational algorithms for unsupervised noise removal in high-resolution EM datasets.

### 2. Neuron Reconstruction Pipeline
* **`Segmentation`**
    * Deep learning models for the dense segmentation of zebrafish neurons. This module handles the voxel-wise classification required to isolate individual neuronal structures from background tissue.
* **`Aggregate`**
    * **Neuron Aggregation Algorithms:** A set of post-processing methods used to aggregate over-segmented fragments or probabilistic maps into coherent neuron proposals during the reconstruction process.
* **`Merge_block`**
    * **Block-wise Stitching:** Algorithms designed to solve the large-scale reconstruction challenge. This module handles the seamless stitching and merging of neuron segments that span across different processing blocks (sub-volumes), ensuring global continuity.

### 3. Ultrastructure & Registration
* **`Zeb-Mito-Syn`**
    * **Ultrastructure Recognition:** A specialized detection suite for identifying and segmenting intracellular organelles, specifically mitochondria and synapses, within the zebrafish EM volume.
* **`Correlative Light and Electron Microscopy` (CLEM)**
    * **Registration Framework:** Robust algorithms for the cross-modal alignment of Light Microscopy (LM) and Electron Microscopy (EM) datasets, enabling precise overlay of functional signals onto structural data.

### 4. Analysis & Quality Control
* **`Quality Assessment`**
    * **Automated IQA:** Algorithms for the objective assessment of image quality in zebrafish EM acquisitions. This module helps identify artifacts, blur, or contrast issues in raw data before processing.
* **`Zfish-Nc-cluster-all`**
    * **Clustering & Analysis:** A comprehensive analysis package for the final reconstruction results. It includes tools for feature extraction, neuronal clustering, and statistical evaluation of the connectome data.

## Usage
* [Add instructions on how to run the code here]

## Citation
If you use this code or models in your research, please cite:
* [Your Name/Paper Title Here]

## License
[Your License Name, e.g., MIT License]
