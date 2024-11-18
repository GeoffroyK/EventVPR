# Visual Place Recognition with Event-Based Data
Visual Place Recognition with Spiking Neural Networks and input datastream from DVS.
This repository contains the implementation of a Visual Place Recognition (VPR) system using event-based data.

## Overview

Event-based cameras, also known as neuromorphic cameras, capture changes in the scene at microsecond resolutions, making them ideal for scenarios with high-speed motion or challenging lighting conditions. This project implements and evaluates a VPR pipeline specifically tailored for event-based data, showcasing the potential of event-based cameras in robotics and computer vision tasks.

### Key Features

- **Robust to dynamic lighting conditions**: Event cameras are inherently HDR, enabling VPR in diverse environments.
- **Efficient processing**: Exploits the sparse nature of event data for computational efficiency.
- **Evaluation on real-world dataset**: Uses the Brisbane Event VPR Dataset and later DD20

## Datasets

The Brisbane Event VPR Dataset is a publicly available dataset curated for testing VPR algorithms with event-based data. It provides event streams captured in various indoor and outdoor locations in Brisbane, Australia.  

### Dataset Features

- Event-based camera recordings with timestamps.
- Frame-based camera recordings
- Ground truth place labels for VPR benchmarking.

### Access the Dataset

The Brisbane Event VPR Dataset is available [here](https://open.qcr.ai/dataset/brisbane_event_vpr_dataset/).

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/visual-place-recognition-event-data.git
cd visual-place-recognition-event-data
```
