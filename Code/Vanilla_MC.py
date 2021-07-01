import numpy as np
import pandas as pd
import math
import networkx as nx
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from timeit import default_timer as timer

"""""
###### Vanilla MC Algorithm ######
"""""
def run(data, l, alpha):
    t_begin = timer()

    G = nx.read_edgelist('data/'+ data + '.txt', create_using=nx.Graph(), nodetype=int)

    t_loadtxt = timer()

    n = G.number_of_nodes()
    # m = G.number_of_edges()

    csr = nx.to_scipy_sparse_matrix(G, nodelist = sorted(G.nodes()), format='csr')

    t_csr = timer()

    r = alpha*n

    start_nodes = np.repeat(np.arange(0,n,1),r/n)
    threads_per_block = 1024
    blocks_per_grid = math.ceil(r/threads_per_block)
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=np.random.randint(1e5))

    t_rng = timer()

    start_nodes_device = cuda.to_device(start_nodes)
    indptr_device = cuda.to_device(csr.indptr)
    indices_device = cuda.to_device(csr.indices)

    hit_times = cuda.device_array(r)
    sum_hit_times = cuda.device_array(1)
    t_gpu_copy = timer()

    gpu_random_walks[blocks_per_grid, threads_per_block](start_nodes, indptr_device, indices_device, hit_times, sum_hit_times, rng_states, n, r, l)

    cuda.synchronize()
    t_gpu_run = timer()

    sum_hit_times = sum_hit_times.copy_to_host()


    t_cpu_copy = timer()

    Kemeny_constant = n*sum_hit_times[0]/r+n-l-1
    t_end = timer()

    print('Kemeny Constant: ', Kemeny_constant)
    print('Runtime: {} s'.format(t_end - t_csr))
    print('GPU Kernel Runtime: {} s'.format(t_gpu_run - t_gpu_copy))


"""""
###### GPU Kernel ######
"""""
@cuda.jit
def gpu_random_walks(root_nodes, indptr, indices, hit_time, sum_hit_time, rng_states, n, r, l):
    thread_id = cuda.grid(1)
    if thread_id < r:
        hit_time[thread_id] = 0
        curr_node = root_nodes[thread_id]
        for i in range(0, l):
            start_idx = indptr[curr_node]
            end_idx = indptr[curr_node + 1]
            neighbors = indices[start_idx:end_idx]
            rand_float = xoroshiro128p_uniform_float32(rng_states, thread_id)
            choice = int(rand_float * len(neighbors))
            next_node = neighbors[choice]
            if next_node == root_nodes[thread_id]:
                hit_time[thread_id] += 1
            curr_node = next_node
        cuda.atomic.add(sum_hit_time, 0, hit_time[thread_id])



"""""
###### Run the Experiments ######
###### Please change the parameter accordingly ######
"""""
run(data='EmailEnron_scc', l=3124, alpha=100)
