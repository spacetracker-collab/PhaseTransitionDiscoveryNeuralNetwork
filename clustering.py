
import torch
import torch.nn as nn

class ClusterLayer(nn.Module):
    def __init__(self,k=3):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(k,2))

    def forward(self,z):
        q = 1/(1+torch.sum((z.unsqueeze(1)-self.centers)**2,dim=2))
        q = q/torch.sum(q,dim=1,keepdim=True)
        return q
