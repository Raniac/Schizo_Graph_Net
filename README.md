# Schizo_Graph_Net

Discriminative Analysis of Schizophrenia Using Graph Neural Network.

## Environment and Dependencies

All dependencies are covered by a Docker image named ```imcomking/pytorch_geometric```.
```bash
$ nvidia-docker run -it --rm -v /home/bennyray/Projects/neuro-learn/local/neuro-learn-local/dev/schizo_graph_net/:/workspace/schizo_graph_net/ imcomking/pytorch_geometric /bin/bash
root@d0f1e42bb63f:/workspace$ cd schizo_graph_net
root@d0f1e42bb63f:/workspace/schizo_graph_net$ python main.py
```

## Introduction

The code in this repository provides the implementation of graph-neural-network-based discriminative analysis of Schizophrenia with brain connectivity network data.

## Contact

Should you encouter any problems, please feel free to file an issue or contact leibingye@outlook.com.