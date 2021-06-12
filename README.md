[![](https://img.shields.io/badge/license-GPL--3.0-blue)](https://www.gnu.org/licenses/)
[![](https://img.shields.io/badge/Python-3.7.10-green)](https://www.python.org/downloads/release/python-3710/)

# An Efficient and Scalable Algorithm for Estimating Kemeny's Constant of a Markov Chain on Large Graphs

<!---This is a C++, Python3 implementation of our Monte Carlo Algorithm algorithm for the task of estimating Kemeny's Constant of a Markov Chain on large graphs, as described in our paper.--> 

## Overview

Here we provide the GPU implementation of our Monte Carlo (MC) algorithm used in our paper in ACM KDD 2021. 
The repository is organised as follows:
* ```requirements.txt```: All required Python libraries to run the code.
* ```Code/Vanilla_MC.py```: GPU implementation of vanilla MC algorithm.
* ```Code/Dynamic_MC.py```: GPU implementation of dynamic MC algorithm.


## Requirements
<!---numpy==1.19.5--> 
<!---pandas=1.1.5--> 
<!---numba=0.51.2--> 
<!---networkx=2.5.1--> 

[![](https://img.shields.io/badge/numpy-1.19.5-green)](https://numpy.org/devdocs/index.html)
[![](https://img.shields.io/badge/pandas-1.1.5-green)](https://pandas.pydata.org/pandas-docs/stable/index.html)
[![](https://img.shields.io/badge/numba-0.51.2-green)](http://numba.pydata.org/)
[![](https://img.shields.io/badge/networkx-2.5.1-green)](https://networkx.org/)

```bash
$ pip install -r requirements.txt
```

In addition, [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) has been used.


## Dataset
We use 13 real-world undirected network datasets:
| |# Nodes (n) | # Edges (m) | 
| ---------- | :-----------:  | :-----------: |
|HEP-TH| 8638 | 24806 |
|Astro-ph| 17903 | 196972 |
|CAIDA| 26475 | 53381 |
|EmailEnron| 33696 | 180811 |
|Brightkite| 56739 | 212945 |
|wiki-Talk| 92117 | 360767 |
|Gowalla| 196591 | 950327 |
|com-DBLP| 317080 | 1049866 |
|Amazon| 334863 | 925872 |
|soc-flickr| 513969 | 3190452 |
|soc-digg| 770799 | 5907132 |
|Youtube| 1134890 | 2987624 |
|Skitter| 1694616 | 11094209 |

The datasets are downloaded form:
1. [SNAP](http://snap.stanford.edu/data/index.html)
2. [Network Repository](http://networkrepository.com/)

> Note: 
> 1. The statistics shown above are for the largest strongly connected components (LSCC) of graphs.
> 2. You can also use your own network datasets.

## Algorithms

You can choose between the following algorithms: 
* Vanilla MC algorithm
* Dynamic MC algorithm

## Running the code
1. Download the data as described above.
2. Run ```Code/Vanilla_MC.py``` to compute Kemeny's Constant by vanilla MC algorithm.
3. Run ```Code/Dynamic_MC.py``` to compute Kemeny's Constant by dynamic MC algorithm.

> Note:
> 1. To reproduce our results, please download the same set of data as we used in the paper.  
> 2. For different test cases, please change the settings as described in each code file.

<!---## Cite--> 

<!---Please cite our paper if you use this code in your own work:--> 
