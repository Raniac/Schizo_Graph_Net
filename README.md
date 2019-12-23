# Schizo_Graph_Net

Discriminative Analysis of Schizophrenia Using Graph Neural Network.

## Environment and Dependencies

All covered by a Docker image named ```imcomking/pytorch_geometric```.
```bash
$ nvidia-docker run -it --rm -v /home/bennyray/Projects/neuro-learn/local/neuro-learn-local/dev/schizo_graph_net/:/workspace/schizo_graph_net/ imcomking/pytorch_geometric /bin/bash
root@d0f1e42bb63f:/workspace$ cd schizo_graph_net
root@d0f1e42bb63f:/workspace/schizo_graph_net$ python main.py
```