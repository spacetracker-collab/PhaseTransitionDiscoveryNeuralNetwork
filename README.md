
# Advanced Phase Discovery with Molecular Dynamics + GNN

## Overview
This project implements:

- Lennard-Jones molecular dynamics simulation
- Graph Neural Network (GNN) for particle-level learning
- Autoencoder + Deep Clustering
- Automatic phase transition detection
- Approximate critical point detection

## Features
- Real particle simulation (not synthetic Gaussian blobs)
- Graph-based representation of particles
- Emergent phase detection from structure
- Detects transitions via cluster instability

## Install
pip install numpy matplotlib torch scikit-learn networkx

## Run
python main.py

## Files
- md_simulation.py → Lennard-Jones simulation
- gnn_model.py → Graph Neural Network
- clustering.py → Deep clustering logic
- main.py → pipeline
- notebook.ipynb → Colab version

## Output
- Latent phase clusters
- Phase transition curve
- Critical region estimation

