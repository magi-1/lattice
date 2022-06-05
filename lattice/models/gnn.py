from lattice.features import FeatureDict, NodeFeature, EdgeFeature

import jax
import jraph
import haiku as hk

import numpy as np
from typing import Union


def parse_node_features(features: FeatureDict) -> Union[np.ndarray, None]:
    values = [v for feat, v in features.items() if isinstance(feat, NodeFeature)]
    if values:
        return np.concatenate(values).T


def parse_edge_features(features: FeatureDict) -> Union[np.ndarray, None]:
    values = [
        v.flatten() for feat, v in features.items() if isinstance(feat, EdgeFeature)
    ]
    if values:
        return np.array(values).T


def construct_graph(
    features: FeatureDict, global_features: dict = None
) -> jraph.GraphsTuple:

    # graph features
    node_features = parse_node_features(features)
    edge_features = parse_edge_features(features)

    # graph structure
    n_nodes = node_features.shape[0]
    n_edges = int(0.5 * n_nodes * (n_nodes + 1))
    edges = [(i, j) for i in range(n_nodes) for j in range(n_nodes)]

    return jraph.GraphsTuple(
        n_node=np.asarray([n_nodes]),
        n_edge=np.asarray([n_edges]),
        nodes=node_features,
        edges=edge_features,
        globals=global_features,
        senders=np.asarray([e[0] for e in edges]),
        receivers=np.asarray([e[1] for e in edges]),
    )


def network_definition(
    graph: jraph.GraphsTuple,
    embedding_dim: int = 16,
    hidden_dim: int = 16,
    message_pasing_steps: int = 2,
) -> jraph.ArrayTree:

    embedding = jraph.GraphMapFeatures(
        embed_edge_fn=jax.vmap(hk.Linear(output_size=embedding_dim)),
        embed_node_fn=jax.vmap(hk.Linear(output_size=embedding_dim)),
        embed_global_fn=jax.vmap(hk.Linear(output_size=embedding_dim)),
    )
    graph = embedding(graph)

    @jax.vmap
    @jraph.concatenated_args
    def update_fn(features):
        net = hk.Sequential(
            [
                hk.Linear(hidden_dim),
                jax.nn.relu,
                hk.Linear(hidden_dim),
                jax.nn.relu,
                hk.Linear(hidden_dim),
                jax.nn.relu,
            ]
        )
        return net(features)

    for _ in range(message_pasing_steps):
        gn = jraph.GraphNetwork(
            update_edge_fn=update_fn,
            update_node_fn=update_fn,
            update_global_fn=update_fn,
        )
        graph = gn(graph)

    return hk.Linear(3)(graph.nodes)
