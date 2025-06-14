import networkx as nx
from pyvis.network import Network
from .parser import parse_functions
import os


def build_pipeline_graph(source_dir: str, output_html: str) -> None:
    calls = parse_functions(source_dir)
    G = nx.DiGraph()
    for func, called_set in calls.items():
        G.add_node(func)
        for callee in called_set:
            G.add_edge(func, callee)
    net = Network(directed=True, height='750px', width='100%')
    net.from_nx(G)
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    net.write_html(output_html)
