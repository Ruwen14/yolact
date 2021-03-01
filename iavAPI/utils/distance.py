import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import torch
from torch import Tensor

def pairwise_Euclidean_VECTOR(A: np.ndarray, B: np.ndarray, mode='CPU'):
    assert (mode == 'CPU' or mode == 'GPU'), 'choose mode'

    if mode == 'CPU':
        distance_matrix = cdist(A, B)
        return distance_matrix

    if mode == 'GPU':
        Tensor_A = torch.FloatTensor(A)
        Tensor_B = torch.FloatTensor(B)
        distance = torch.cdist(Tensor_A, Tensor_B)
        distance_on_cpu = distance.cpu().detach().numpy()
        return distance_on_cpu

def pairwise_minEuclidean_VECTOR(A: np.ndarray, B: np.ndarray, mode='CPU'):
    assert (mode=='CPU' or mode=='GPU'), 'choose mode'

    if mode=='CPU':
        distance_matrix = cdist(A, B)
        rel_dist_minima = distance_matrix.min(axis=1)
        abs_dist_min = rel_dist_minima.min()
        abs_dist_min_idx = np.argmin(rel_dist_minima)
        min_idx = np.argmin(distance_matrix, axis=1)[abs_dist_min_idx]

        return (A[abs_dist_min_idx], B[min_idx]), abs_dist_min

    if mode == 'GPU':
        assert torch.cuda.current_device() == 0, 'Cant use "mode==GPU" without GPU'

        TENSOR_A = torch.FloatTensor(A)
        TENSOR_B = torch.FloatTensor(B)
        TENSOR_Distance = torch.cdist(TENSOR_A, TENSOR_B)
        mins, indices = torch.min(TENSOR_Distance, dim=1)
        TENSOR_min_distance_abs = torch.min(mins)

        TENSOR_closestA = TENSOR_A[torch.argmin(mins)]
        TENSOR_closestB = TENSOR_B[torch.argmin(TENSOR_Distance, dim=1)[torch.argmin(mins)]]

        ARRAY_closestA = TENSOR_closestA.cpu().detach().numpy()
        ARRAY_closestB = TENSOR_closestB.cpu().detach().numpy()
        closest_Points = (tuple(ARRAY_closestA.astype(int)), tuple(ARRAY_closestB.astype(int)))
        FLOAT32_min_distance_abs = TENSOR_min_distance_abs.cpu().detach().numpy()

        return closest_Points, float(FLOAT32_min_distance_abs)


def pairwise_minEuclidean_MATRIX(VECTOR_A, MATRIX_B, mode='CPU'):
    if not (mode=='GPU' or mode=='CPU' or mode=='numba' or mode=='tree'):
        mode='CPU'; print("Force mode -> CPU")

    # Slow! 566ms @12500 points, 1.88 s @12500000 points
    if mode =='numba':
        from numba import jit # will be cached no worries

        @jit(nopython=True, fastmath=True)
        def euclidean(u, v):
            n = len(v)
            dist = np.zeros((n))
            for i in range(n):
                dist[i] = np.sqrt(np.nansum((u - v[i]) ** 2))
            return dist

        @jit(nopython=True, fastmath=True)
        def vector_to_matrix_distance(u, m):
            m_rows = m.shape[0]
            m_cols = m.shape[1]
            u_rows = u.shape[0]

            out_matrix = np.zeros((m_rows, u_rows, m_cols))
            for i in range(m_rows):
                for j in range(u_rows):
                    out_matrix[i][j] = euclidean(u[j], m[i])

            return out_matrix


        return vector_to_matrix_distance(VECTOR_A,MATRIX_B)

    # 119 μs @12500 points
    # 96 ms @12500000 points
    if mode =='CPU':
        # Fast cdist function
        # List compression for speed up
        distance_matrix = np.array([cdist(VECTOR_A  , MATRIX_B[i])
                                    for i in range(len(MATRIX_B))])
        return distance_matrix

    # 351 μs @12500 points
    # 47 ms @12500000 points
    if mode =='GPU':
        Tensor_A = torch.FloatTensor(VECTOR_A)
        Tensor_B = torch.FloatTensor(MATRIX_B)

        distance_matrix =np.array([torch.cdist(Tensor_A, Tensor_B[i]).cpu().detach().numpy()
                                    for i in range(len(Tensor_B))])

        return distance_matrix

    if mode =='tree':
        dist = np.array([cKDTree(vec).query(VECTOR_A) for vec in (MATRIX_B)])

        return dist


def nearestNodes_top_K(node, nodes, top_k: int = 1):
    assert isinstance(top_k, int), "top_k has to be an integer"

    if not isinstance(node, np.ndarray):
        node = np.asarray(node)

    if not isinstance(nodes, np.ndarray):
        nodes = np.asarray(nodes)


    closest_idx = cdist(node, nodes)
    closest_index_top_k = list(np.argsort(closest_idx[0])[:top_k])

    return nodes[closest_index_top_k]


def cosine_similarity_GPU(a: Tensor, b: Tensor):

    """
     Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
     :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
     """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)

    return torch.mm(a_norm, b_norm.transpose(0, 1))

def k_d_tree():
    pass

def sort_by_distance(ListA, ListB, returnType='List'):
    """
    Args:
        ListA:
        ListB:

    Returns:
        sorted ListB, so that each entry x --> ListB[x] has least distance to entry x of ListA[x]
    """
    distance_matrix = cdist(ListA, ListB)

    minima_indices = [np.argmin(vector) for vector in distance_matrix]

    if returnType == 'List':
        # Sort ListB by Indice of Minima
        sorted_ListB = [ListB[indice] for indice in minima_indices]
        return sorted_ListB

    else:
        return minima_indices