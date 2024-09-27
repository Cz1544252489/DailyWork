import os
import torch
os.environ["OMP_NUM_THREADS"] = "1"
from Clustering_aux import initial, get_val
from sklearn.cluster import KMeans
from torch import nn
from geoopt.manifolds import Stiefel, Euclidean
from torch.optim import Adam, SGD



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Theta_bbc, F_bbc, Theta_guardian, F_guardian = initial(backend="torch", device = device)

labels = ['business', 'entertainment', 'health', 'politics', 'sport', 'tech']

F = F_bbc
Theta = Theta_bbc
F_star = F_guardian
Theta_star = Theta_guardian


# bbc 作为 star 视角
class Model(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.F = nn.Parameter(F)

    def forward(self, F_star, Theta, lambda_r):
        term1 = torch.trace(self.F.T @ Theta @ self.F)
        term2 = lambda_r * torch.trace(self.F @ self.F.T @ F_star @ F_star.T)
        return term1 + term2

#
class Model_star(nn.Module):
    def __init__(self, F_star):
        super().__init__()
        self.F_star = nn.Parameter(F_star)

    def forward(self, F, lambda_r):
        term = lambda_r * torch.trace(F @ F.T @ self.F_star @ self.F_star.T)
        return term


lambda_r = 1

T = 200
epsilon = 1e-6
learning_rate = 0.01

model = Model(F= F)
optim = Adam(model.parameters(), lr = learning_rate)
model_star = Model_star(F_star = F_star)
optim_star = Adam(model_star.parameters(), lr = learning_rate)

for j in range(T):
    if j > 0:
        model.zero_grad()
    output = model(F_star, Theta, lambda_r)
    output.backward()
    optim.step()
    norm_of_grad1 = torch.norm(model.F.grad)
    if norm_of_grad1.item() < epsilon:
        break
    val = get_val(model_star.F_star, model.F, lambda_r)
    print([j, output.item(), norm_of_grad1.item(), val.item(), "model"])


for i in range(T):
    if i>0:
        model_star.zero_grad()
    output_star = model_star(F, lambda_r)
    output_star.backward()
    optim_star.step()
    norm_of_grad = torch.norm(model_star.F_star.grad)
    if norm_of_grad.item() < epsilon:
        break
    val = get_val(model_star.F_star, model.F, lambda_r)
    print([i, output_star.item(), norm_of_grad.item(), val.item()])










test_F = F_star.to('cpu')
test_F = test_F.clone()
test_F = test_F.numpy()

# 开始聚类
K = 6
kmeans = KMeans(n_clusters=K, random_state=0)
result = kmeans.fit(test_F)
#
# # 打印聚类中心
# print("Cluster centers:")
# print(kmeans.cluster_centers_)
#
#
print("aa")