FROM imcomking/pytorch_geometric

WORKDIR /schizo_graph_net
ADD . /schizo_graph_net/

CMD ["python", "/schizo_graph_net/main.py"]
