
import torch
import numpy as np
import matplotlib.pyplot as plt
from md_simulation import simulate
from gnn_model import SimpleGNN
from clustering import ClusterLayer
from sklearn.cluster import KMeans

traj = simulate()
data = traj.reshape(-1,2)
data = torch.tensor(data,dtype=torch.float32)

model = SimpleGNN()
cluster = ClusterLayer()

opt = torch.optim.Adam(list(model.parameters())+list(cluster.parameters()), lr=0.01)

# Pretrain
for _ in range(50):
    opt.zero_grad()
    z = model(data)
    loss = torch.mean(z**2)
    loss.backward()
    opt.step()

# init clusters
with torch.no_grad():
    z = model(data)
kmeans = KMeans(n_clusters=3).fit(z.numpy())
cluster.centers.data = torch.tensor(kmeans.cluster_centers_,dtype=torch.float32)

# train clustering
for epoch in range(100):
    opt.zero_grad()
    z = model(data)
    q = cluster(z)
    loss = torch.mean(q*torch.log(q+1e-8))
    loss.backward()
    opt.step()

# output
with torch.no_grad():
    z = model(data)
    q = cluster(z)
    labels = torch.argmax(q,1).numpy()

z = z.numpy()

plt.scatter(z[:,0],z[:,1],c=labels,s=2)
plt.title("Phase Clusters from MD + GNN")
plt.show()
