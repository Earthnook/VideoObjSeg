""" A utility to visualize compute graph of a model
Code from: https://discuss.pytorch.org/t/print-autograd-graph/692/5
"""
from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '('+(', ').join(['%d'% v for v in var.size()])+')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.creator)
    return dot

def make_graph(model, input_dim):
    """  make the compute graph of a model, whose output and input are two one-variable
    @ Args:
        model: a nn.Module instance
        input_dim: a tuple specifying the input variable dimension
    """
    inputs = torch.randn(1, *input_dim)
    outputs = model(Variable(inputs))
    graph = make_dot(outputs)

    return graph


if __name__ == "__main__":
    # an example
    inputs = torch.randn(1,3,224,224)
    resnet18 = models.resnet18()
    y = resnet18(Variable(inputs))
    print(y)

    g = make_dot(y)
    # Now g is the compute graph 
