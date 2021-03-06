# Schizo_Graph_Net

Discriminative Analysis of Schizophrenia Using Graph Neural Network.

## Environment and Dependencies

All dependencies are covered by a Docker image named ```imcomking/pytorch_geometric```.
```bash
$ nvidia-docker run -it --rm -v /home/bennyray/Projects/neuro-learn/sgn/:/workspace/ imcomking/pytorch_geometric /bin/bash
root@d0f1e42bb63f:/workspace$ python main.py
```

## Introduction

The code in this repository provides the implementation of graph-neural-network-based discriminative analysis of Schizophrenia with brain connectivity network data.

## Dockerized SGN

We have dockerized SGN as a service (see [NEURO-LEARN-DOCKER-SGN](https://github.com/Raniac/NEURO-LEARN-DOCKER-SGN)), which can be utilized by [NEURO-LEARN](https://github.com/Raniac/NEURO-LEARN). Create a new deep learning task with NEURO-LEARN and see!

## Contact

Should you encouter any problems, please feel free to file an issue or contact leibingye@outlook.com.