import numpy as np
import pandas as pd
import math
import networkx as nx
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from timeit import default_timer as timer

"""""
###### Dynamic MC Algorithm ######
"""""
def run(data, l, alpha, epsilon):
    G = nx.read_edgelist('data/'+ data + '.txt', create_using=nx.Graph(), nodetype=int)
    n = G.number_of_nodes()

    csr = nx.to_scipy_sparse_matrix(G, nodelist = sorted(G.nodes()), format='csr')

    t_start = timer()

    r = alpha*n

    initial_start_nodes = np.repeat(np.arange(0,n,1),r/n)
    threads_per_block = 1024
    blocks_per_grid = math.ceil(r/threads_per_block)
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=np.random.randint(1e5))

    initial_start_nodes_device = cuda.to_device(initial_start_nodes)
    start_nodes_device = cuda.to_device(initial_start_nodes)
    indptr_device = cuda.to_device(csr.indptr)
    indices_device = cuda.to_device(csr.indices)

    hit_times = cuda.device_array(r)
    sum_hit_times = cuda.device_array(1)

    t_gpu_kernel_start_0 = timer()

    gpu_random_walks[blocks_per_grid, threads_per_block](initial_start_nodes_device, start_nodes_device, indptr_device, indices_device, hit_times, sum_hit_times, rng_states, n, r, l)

    cuda.synchronize()

    t_gpu_kernel_end_0 = timer()
    t_gpu_kernel = t_gpu_kernel_end_0 - t_gpu_kernel_start_0

    sum_hit_times_host = sum_hit_times.copy_to_host()

    Kemeny_constant = n*sum_hit_times_host[0]/r+n-l-1
    Kemeny = [Kemeny_constant]

    dl = l
    EPS = epsilon*n
    diff = EPS
    while diff >= EPS:
        l += dl
        t_gpu_kernel_start_1 = timer()

        gpu_random_walks[blocks_per_grid, threads_per_block](initial_start_nodes_device, start_nodes_device, indptr_device, indices_device, hit_times, sum_hit_times, rng_states, n, r, dl)
        cuda.synchronize()
        t_gpu_kernel_end_1 = timer()
        t_gpu_kernel += t_gpu_kernel_end_1-t_gpu_kernel_start_1

        sum_hit_times_host = sum_hit_times.copy_to_host()
        Kemeny_constant_ = n*sum_hit_times_host[0]/r+n-l-1

        diff = abs(Kemeny_constant - Kemeny_constant_)

        Kemeny_constant = Kemeny_constant_
        Kemeny.append(Kemeny_constant)

    t_end = timer()

    print('Kemeny Constant: ', Kemeny[-1])
    print('Runtime: {} s'.format(t_end - t_start))
    print('GPU Kernel Runtime: {} s'.format(t_gpu_kernel))

"""""
###### GPU Kernel ######
"""""
@cuda.jit
def gpu_random_walks(initial_start_nodes, start_nodes, indptr, indices, hitting, sum_hitting, rng_states, n, r, l):
    thread_id = cuda.grid(1)
    if thread_id < r:
        curr_node = start_nodes[thread_id]
        hitting[thread_id] = 0
        for i in range(0, l):
            start_idx = indptr[curr_node]
            end_idx = indptr[curr_node + 1]
            neighbors = indices[start_idx:end_idx]
            rand_float = xoroshiro128p_uniform_float32(rng_states, thread_id)
            choice = int(rand_float * len(neighbors))
            next_node = neighbors[choice]
            if next_node == initial_start_nodes[thread_id]:
                hitting[thread_id] += 1
            curr_node = next_node
        start_nodes[thread_id] = curr_node
        cuda.atomic.add(sum_hitting, 0, hitting[thread_id])



"""""
###### Run the Experiments ######
###### Please change the parameter accordingly ######
"""""

run(data='EmailEnron_scc', l=200, alpha=100, epsilon=0.0005)
