# RADIANT3D

**Reliability-Guided Adaptive LiDAR–Camera Fusion for 3D Semantic Segmentation**

## Overview
RADIANT3D investigates reliability-guided fusion of LiDAR and RGB camera features for 3D semantic segmentation.

The approach uses local LiDAR point density and point-to-sensor distance as reliability signals to guide camera feature injection into LiDAR representations.

## Research Details
- Baseline: Perception-Aware Multi-Sensor Fusion (PMF), ICCV 2021.
- Dataset: SemanticKITTI-FOV.
- Framework: PyTorch.
- Architecture: RADIANT3DNet.
- Scope: Software-based research; no hardware implementation.

## Repository Contents
- Model architectures and fusion modules.
- Training and validation scripts.
- SemanticKITTI data-loading and preprocessing code.
- Experiment configurations and comparison script.

## Experiments
The repository preserves the historical 15-epoch AdaptivePMF V1 configuration separately from the 50-epoch RADIANT3D configuration.

The historical AdaptivePMF V1 experiment achieved 58.42% validation mIoU. This result must not be interpreted as an independently trained RADIANT3DNet result.

## Dataset and Checkpoints
SemanticKITTI-FOV data and model checkpoints are not included. Configure their local paths before running experiments.

## Acknowledgements
This research builds upon the PMF multi-sensor fusion framework.
