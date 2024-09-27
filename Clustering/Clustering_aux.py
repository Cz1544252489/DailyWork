import pandas as pd
from scipy.io import mmread
import os
import numpy as np
import torch
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sympy.physics.quantum.matrixutils import sparse


def load_csr_matrix(file_path):
    sparse_matrix_coo = mmread(file_path)
    return csr_matrix(sparse_matrix_coo)

def get_data():
    # 设置根路径
    root_path = "E:\\DL_datasets\\3sources\\"

    # 新闻源和文件类型
    sources = ['bbc', 'guardian', 'reuters']
    file_types = ['mtx', 'terms', 'docs']

    data = {}
    # 使用循环创建变量并加载数据
    for source in sources:
        for ftype in file_types:
            filename = f"3sources_{source}.{ftype}"
            filepath = os.path.join(root_path, filename)

            # 根据文件类型选择读取方法
            if ftype == 'mtx':
                data[f"{source}_{ftype}"] = load_csr_matrix(filepath)
            else:
                data[f"{source}_{ftype}"] = pd.read_csv(filepath, header=None, names=[ftype.capitalize()])

    # 调用函数并打印结果
    disjiont_clist_filename = "3sources.disjoint.clist"
    disjiont_clist_path = os.path.join(root_path, disjiont_clist_filename)
    overlap_clist_filename = "3sources.overlap.clist"
    overlap_clist_path = os.path.join(root_path, overlap_clist_filename)

    data['disjoint_clist'] = load_data_to_dataframe(disjiont_clist_path)
    data['overlap_clist'] = load_data_to_dataframe(overlap_clist_path)

    return data

def load_data_to_dataframe(filename):
    data_list = []
    with open(filename, 'r') as file:
        for line in file:
            category, ids = line.strip().split(': ')
            for id in ids.split(','):
                data_list.append({'category': category, 'id': int(id)})

    # 创建DataFrame
    df = pd.DataFrame(data_list)
    return df

# 构建关联矩阵H
def get_Theta_and_F(sample, docs, clist, labels):
    n = len(sample)
    m = len(labels)
    H = np.zeros([n, m])
    for article in docs.values.reshape([-1]):
        for label, article_real in clist.values:
            if article == article_real:
                # print([np.where(bbc_docs.values.reshape([-1])==article), labels.index(label)])
                H[np.where(sample==article), labels.index(label)] = 1

    # 构建权重W
    W = np.diag(np.random.rand(m))

    # 构建超边和超顶点的度矩阵
    D_e = np.zeros(m)
    for i in range(m):
        D_e[i] = sum(H[:,i])

    D_e = np.diag(D_e)

    D_v = np.zeros(n)
    for i in range(n):
        D_v[i] = sum(H[i,:])

    D_v = np.diag(D_v)

    D_v_inv_sqrt = np.linalg.inv(np.sqrt(D_v))
    D_e_inv = np.linalg.inv(D_e)
    Theta = D_v_inv_sqrt @ H @ W @ D_e_inv @ H.T @ D_v_inv_sqrt

    _, F = eigsh(Theta, m, which='LM')

    return Theta, F

def initial(flag=1, backend="numpy", clist_type="overlap",*,device):
    # 按照文件名生成数据
    data = get_data()
    for key, value in data.items():
        globals()[key] = value

    labels = ['business', 'entertainment', 'health', 'politics', 'sport', 'tech']

    S_bbc = set(bbc_docs.values.reshape([-1]))
    S_guardian = set(guardian_docs.values.reshape([-1]))
    S_reuters = set(reuters_docs.values.reshape([-1]))

    match flag:
        case 0:
            sample = list(S_bbc & S_guardian & S_reuters)
        case 1:
            sample = list(S_bbc & S_guardian)
        case 2:
            sample = list(S_bbc & S_reuters)
        case 3:
            sample = list(S_reuters & S_guardian)
        case _:
            raise ValueError("输入flag错误")

    # 两个视角
    # bbc & guardian
    if clist_type == "disjoint":
        clist = disjoint_clist
    elif clist_type == "overlap":
        clist = overlap_clist
    else:
        raise ValueError("clist_type wrong!")

    Theta_bbc, F_bbc = get_Theta_and_F(sample, bbc_docs, disjoint_clist, labels)
    Theta_guardian, F_guardian = get_Theta_and_F(sample, guardian_docs, disjoint_clist, labels)

    if backend == "torch":
        Theta_bbc = torch.from_numpy(Theta_bbc)
        F_bbc = torch.from_numpy(F_bbc)
        Theta_guardian = torch.from_numpy(Theta_guardian)
        F_guardian = torch.from_numpy(F_guardian)

        Theta_bbc = Theta_bbc.to(device)
        F_bbc = F_bbc.to(device)
        Theta_guardian = Theta_guardian.to(device)
        F_guardian = F_guardian.to(device)

    return Theta_bbc, F_bbc, Theta_guardian, F_guardian

def initial1(flag=1 , clist_type="disjoint"):
    # 按照文件名生成数据
    data = get_data()
    for key, value in data.items():
        globals()[key] = value

    labels = ['business', 'entertainment', 'health', 'politics', 'sport', 'tech']

    S_bbc = set(bbc_docs.values.reshape([-1]))
    S_guardian = set(guardian_docs.values.reshape([-1]))
    S_reuters = set(reuters_docs.values.reshape([-1]))

    match flag:
        case 0:
            sample = list(S_bbc & S_guardian & S_reuters)
        case 1:
            sample = list(S_bbc & S_guardian)
        case 2:
            sample = list(S_bbc & S_reuters)
        case 3:
            sample = list(S_reuters & S_guardian)
        case _:
            raise ValueError("输入flag错误")

    # 两个视角
    # bbc & guardian
    if clist_type == "disjoint":
        clist = disjoint_clist
    elif clist_type == "overlap":
        clist = overlap_clist
    else:
        raise ValueError("clist_type wrong!")

    Theta_bbc, F_bbc = get_Theta_and_F(sample, bbc_docs, clist, labels)
    Theta_guardian, F_guardian = get_Theta_and_F(sample, guardian_docs, clist, labels)

    return sample

def get_val(F_star, F, lambda_r, flag = 0):
    if flag == 1:
        term = torch.trace(F_star.T @ (torch.eye(F.shape[0])-lambda_r* F @ F.T) @ F_star)
    else:
        term = torch.trace(F.T @ (torch.eye(F_star.shape[0]) - lambda_r * F_star @ F_star.T) @ F)
    return term

def show_high_dimension_result(F, labels):
    pca = PCA(n_components=2)
    X = pca.fit_transform(F)

    plt.figure(figsize=(8, 6))
    show_result(X, labels)

def show_result(X, labels):
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    # cmap的选择还有'rainbow', 'jet'等
    plt.title('PCA of 6D data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter)
    plt.show()

def cluster_and_show(F, num_type, flag ="normal"):
    kmeans = KMeans(n_clusters = num_type, random_state=0)
    spectral = SpectralClustering(n_clusters=num_type, affinity='nearest_neighbors',
                                  assign_labels='kmeans', random_state=42)
    if flag == "spectral":
        labels = spectral.fit_predict(F)
    elif flag == "normal":
        labels = kmeans.fit_predict(F)
    else:
        raise ValueError("Wrong Input!")

    show_high_dimension_result(F, labels)

    return labels



def create_hyperedges(adjacency_matrix, num_hyperedges=6):
    # GPT生成
    num_vertices = adjacency_matrix.shape[0]
    vertices = list(range(num_vertices))
    selected_vertices = set()
    hyperedges = []

    # 计算每个超边应有的节点数量
    vertices_per_edge = num_vertices // num_hyperedges

    for _ in range(num_hyperedges):
        # 选择未选择的节点中的一个作为起始点
        remaining_vertices = list(set(vertices) - selected_vertices)
        if not remaining_vertices:
            break
        current_vertex = np.random.choice(remaining_vertices)

        # 基于相似度选择最近的节点
        distances = adjacency_matrix[current_vertex, :]
        sorted_vertices = np.argsort(-distances)  # 相似度最高的排在前面

        hyperedge = []
        for vertex in sorted_vertices:
            if vertex not in selected_vertices and len(hyperedge) < vertices_per_edge:
                hyperedge.append(vertex)
                selected_vertices.add(vertex)

        # 添加到超边列表
        hyperedges.append(hyperedge)

    # 创建输出矩阵
    output_matrix = np.zeros((num_vertices, num_hyperedges), dtype=int)
    for index, edge in enumerate(hyperedges):
        for vertex in edge:
            output_matrix[vertex, index] = 1

    return output_matrix
