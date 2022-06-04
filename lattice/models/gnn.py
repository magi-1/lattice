from lattice.features import FeatureDict, NodeFeature, EdgeFeature

import jax
import jraph
import numpy as np
from typing import Union


def parse_features(
    features: FeatureDict, feature_type: Union[NodeFeature, EdgeFeature]
):
    return [v.T for feat, v in features.items() if isinstance(feat, feature_type)]


def construct_graph(
    features: FeatureDict, global_features: dict = None
) -> jraph.GraphsTuple:

    # graph features
    node_features = parse_features(features, NodeFeature)
    edge_features = parse_features(features, EdgeFeature)

    # graph structure
    n_nodes = len(node_features[0])
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
        gn = jraph.InteractionNetwork(
            update_edge_fn=update_fn,
            update_node_fn=update_fn,
            update_global_fn=update_fn,
            include_sent_messages_in_node_update=True,
        )
        graph = gn(graph)

    # Can generalize this later to allocation categories
    return hk.Linear(1)(graph.nodes)
