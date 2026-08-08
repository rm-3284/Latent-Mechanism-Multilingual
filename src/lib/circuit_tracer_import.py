# ruff: noqa: F401
# Import from circuit-tracer package (installed from GitHub)
from collections import namedtuple
from typing import List, Dict, Optional

import torch

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import Graph, prune_graph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder

Feature = namedtuple("Feature", ["layer", "pos", "feature_idx"])
Intervention = namedtuple('Intervention', ['supernode', 'scaling_factor'])

class InterventionGraph:
    prompt: str
    ordered_nodes: List["Supernode"]
    nodes: Dict[str, "Supernode"]

    def __init__(self, ordered_nodes: List["Supernode"], prompt: str):
        self.ordered_nodes = ordered_nodes
        self.prompt = prompt
        self.nodes = {}

    def initialize_node(self, node, activations):
        self.nodes[node.name] = node
        if node.features:
            node.default_activations = torch.tensor(
                [activations[feature] for feature in node.features]
            )
        else:
            node.default_activations = None

    def set_node_activation_fractions(self, current_activations):
        for node in self.nodes.values():
            if node.features:
                current_node_activation = torch.tensor(
                    [current_activations[feature] for feature in node.features]
                )
                node.activation = (current_node_activation / node.default_activations).mean().item()
            else:
                node.activation = None
            node.intervention = None
            node.replacement_node = None


class Supernode:
    name: str
    activation: float | None
    default_activations: torch.Tensor | None
    children: List["Supernode"]
    intervention: None
    replacement_node: Optional["Supernode"]

    def __init__(
        self,
        name: str,
        features: List[Feature],
        children: List["Supernode"] = [],
        intervention: Optional[str] = None,
        replacement_node: Optional["Supernode"] = None,
    ):
        self.name = name
        self.features = features
        self.activation = None
        self.default_activations = None
        self.children = children
        self.intervention = intervention
        self.replacement_node = replacement_node

    def __repr__(self):
        return f"Node(name={self.name}, activation={self.activation}, children={self.children}, intervention={self.intervention}, replacement_node={self.replacement_node})"
