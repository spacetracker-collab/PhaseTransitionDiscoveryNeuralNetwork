
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


What you’re seeing is actually **very revealing—and also a sign of a limitation in the current model**. Let’s break it down clearly.

---

# 🔍 What the plot shows

* You have **three colored groups (clusters)** → good ✅
* But instead of forming **separate blobs or regions**, they form a **V-shaped linear structure**

This means:

👉 The model has **not learned distinct phase regions**
👉 It has learned a **single continuous manifold** and just split it artificially

---

# 🧠 What this means physically

Instead of discovering:

* Solid (tight cluster)
* Liquid (spread cluster)
* Gas (very dispersed cluster)

The model is seeing:

> “Everything lies on one continuous spectrum”

So it is behaving like:

* A **1D progression of states** (like temperature increasing)
* Rather than **distinct thermodynamic phases**

---

# ⚠️ Why this happened

## 1. No real phase transition in simulation

Your MD setup:

* No temperature sweep
* No pressure control
* No phase boundary crossing

👉 So the system never actually *changes phase*

---

## 2. GNN is too simple

Your current network:

* Just a few linear layers
* No real **interaction modeling between particles**

👉 It’s not capturing:

* Local structure
* Coordination number
* Crystallinity

---

## 3. Loss function collapse

The loss:

```python
loss = torch.mean(q * torch.log(q))
```

This encourages:

* Smooth distributions
* Not sharp cluster separation

👉 So clusters become **linear splits**, not physical regimes

---

## 4. No order parameter

Real phase detection requires something like:

* Density fluctuations
* Radial distribution function
* Energy variance

Your model lacks these → so it defaults to geometry

---

# 📊 Interpretation of the V-shape

The V-shape likely represents:

* One axis = **energy or expansion direction**
* The split = **arbitrary clustering boundary**

So:

* Upper branch → cluster 1
* Lower branch → cluster 2
* Middle region → cluster 3

But these are **not real phases**, just partitions of a continuum.

---

# 🧩 Key insight

👉 Your model has learned a **state trajectory**, not **phase separation**

That’s actually interesting:

> It’s capturing *dynamics*, not *phase structure*

---

# 🚨 Why this is important

This tells you:

> The system currently behaves like a **single-phase system evolving continuously**

No:

* First-order transition
* Discontinuity
* Critical behavior

---

# 🔥 How to fix it (this is the real breakthrough step)

To actually get **phase discovery**, you need:

---

## 1. Temperature sweep (CRITICAL)

Simulate multiple regimes:

```python
for T in [0.1, 0.5, 1.0, 2.0]:
    simulate(T)
```

👉 This creates:

* Solid-like (low T)
* Liquid-like
* Gas-like (high T)

---

## 2. Add physical features

Instead of raw positions, compute:

* Pairwise distances
* Local density
* Radial distribution function (RDF)

---

## 3. Better GNN (real one)

Use message passing:

* Nodes = particles
* Edges = neighbors
* Learn interactions explicitly

---

## 4. Detect phase transition properly

Look for:

* Cluster instability
* Sudden change in latent distribution
* Variance spikes

---

## 5. Use Variational Autoencoder (VAE)

This helps:

* Separate regimes
* Create **discontinuous latent regions**

---

# 🧠 Final interpretation (one line)

> Your current model has learned a **continuous thermodynamic trajectory**, not discrete **phase-separated regimes**—which means the simulation never actually entered different phases.

---

# 🚀 If you want

I can upgrade your code so that:

* You **actually see 3 separate blobs**
* It **detects phase transition boundaries automatically**
* It even **plots a phase diagram**

Just say:

👉 *“fix it to produce real phase separation”*

