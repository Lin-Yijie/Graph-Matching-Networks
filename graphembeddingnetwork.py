import torch
import torch.nn as nn
from segment import unsorted_segment_sum


class GraphEncoder(nn.Module):
    """Encoder module that projects node and edge features to some embeddings."""

    def __init__(self,
                 node_feature_dim,
                 edge_feature_dim,
                 node_hidden_sizes=None,
                 edge_hidden_sizes=None,
                 name='graph-encoder'):
        """Constructor.

        Args:
          node_hidden_sizes: if provided should be a list of ints, hidden sizes of
            node encoder network, the last element is the size of the node outputs.
            If not provided, node features will pass through as is.
          edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
            edge encoder network, the last element is the size of the edge outptus.
            If not provided, edge features will pass through as is.
          name: name of this module.
        """
        super(GraphEncoder, self).__init__()

        # this also handles the case of an empty list
        self._node_feature_dim = node_feature_dim
        self._edge_feature_dim = edge_feature_dim
        self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
        self._edge_hidden_sizes = edge_hidden_sizes
        self._build_model()

    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self._node_feature_dim, self._node_hidden_sizes[0]))
        for i in range(1, len(self._node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
        self.MLP1 = nn.Sequential(*layer)

        if self._edge_hidden_sizes is not None:
            layer = []
            layer.append(nn.Linear(self._edge_feature_dim, self._edge_hidden_sizes[0]))
            for i in range(1, len(self._edge_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
            self.MLP2 = nn.Sequential(*layer)
        else:
            self.MLP2 = None

    def forward(self, node_features, edge_features=None):
        """Encode node and edge features.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: if provided, should be [n_edges, edge_feat_dim] float
            tensor.

        Returns:
          node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
          edge_outputs: if edge_features is not None and edge_hidden_sizes is not
            None, this is [n_edges, edge_embedding_dim] float tensor, edge
            embeddings; otherwise just the input edge_features.
        """
        if self._node_hidden_sizes is None:
            node_outputs = node_features
        else:
            node_outputs = self.MLP1(node_features)
        if edge_features is None or self._edge_hidden_sizes is None:
            edge_outputs = edge_features
        else:
            edge_outputs = self.MLP2(edge_features)

        return node_outputs, edge_outputs


def graph_prop_once(node_states,
                    from_idx,
                    to_idx,
                    message_net,
                    aggregation_module=None,
                    edge_features=None):
    """One round of propagation (message passing) in a graph.

    Args:
      node_states: [n_nodes, node_state_dim] float tensor, node state vectors, one
        row for each node.
      from_idx: [n_edges] int tensor, index of the from nodes.
      to_idx: [n_edges] int tensor, index of the to nodes.
      message_net: a network that maps concatenated edge inputs to message
        vectors.
      aggregation_module: a module that aggregates messages on edges to aggregated
        messages for each node.  Should be a callable and can be called like the
        following,
        `aggregated_messages = aggregation_module(messages, to_idx, n_nodes)`,
        where messages is [n_edges, edge_message_dim] tensor, to_idx is the index
        of the to nodes, i.e. where each message should go to, and n_nodes is an
        int which is the number of nodes to aggregate into.
      edge_features: if provided, should be a [n_edges, edge_feature_dim] float
        tensor, extra features for each edge.

    Returns:
      aggregated_messages: an [n_nodes, edge_message_dim] float tensor, the
        aggregated messages, one row for each node.
    """
    from_states = node_states[from_idx]
    to_states = node_states[to_idx]
    edge_inputs = [from_states, to_states]

    if edge_features is not None:
        edge_inputs.append(edge_features)

    edge_inputs = torch.cat(edge_inputs, dim=-1)
    messages = message_net(edge_inputs)

    from segment import unsorted_segment_sum
    tensor = unsorted_segment_sum(messages, to_idx, node_states.shape[0])
    return tensor

class GraphPropLayer(nn.Module):
    """Implementation of a graph propagation (message passing) layer."""

    def __init__(self,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,  # int
                 node_hidden_sizes,  # int
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 prop_type='embedding',
                 name='graph-net'):
        """Constructor.

        Args:
          node_state_dim: int, dimensionality of node states.
          edge_hidden_sizes: list of ints, hidden sizes for the edge message
            net, the last element in the list is the size of the message vectors.
          node_hidden_sizes: list of ints, hidden sizes for the node update
            net.
          edge_net_init_scale: initialization scale for the edge networks.  This
            is typically set to a small value such that the gradient does not blow
            up.
          node_update_type: type of node updates, one of {mlp, gru, residual}.
          use_reverse_direction: set to True to also propagate messages in the
            reverse direction.
          reverse_dir_param_different: set to True to have the messages computed
            using a different set of parameters than for the forward direction.
          layer_norm: set to True to use layer normalization in a few places.
          name: name of this module.
        """
        super(GraphPropLayer, self).__init__()

        self._node_state_dim = node_state_dim
        self._edge_state_dim = edge_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes[:]

        # output size is node_state_dim
        self._node_hidden_sizes = node_hidden_sizes[:] + [node_state_dim]
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type

        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different

        self._layer_norm = layer_norm
        self._prop_type = prop_type
        self.build_model()

        if self._layer_norm:
            self.layer_norm1 = nn.LayerNorm()
            self.layer_norm2 = nn.LayerNorm()

    def build_model(self):
        layer = []
        layer.append(nn.Linear(self._node_state_dim*2 + self._edge_state_dim, self._edge_hidden_sizes[0]))
        for i in range(1, len(self._edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
        self._message_net = nn.Sequential(*layer)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            if self._reverse_dir_param_different:
                layer = []
                layer.append(nn.Linear(self._node_state_dim*2 + self._edge_state_dim, self._edge_hidden_sizes[0]))
                for i in range(1, len(self._edge_hidden_sizes)):
                    layer.append(nn.ReLU())
                    layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
                self._reverse_message_net = nn.Sequential(*layer)
            else:
                self._reverse_message_net = self._message_net

        if self._node_update_type == 'gru':
            if self._prop_type == 'embedding':
                self.GRU = torch.nn.GRU(self._node_state_dim * 2, self._node_state_dim)
            elif self._prop_type == 'matching':
                self.GRU = torch.nn.GRU(self._node_state_dim * 3, self._node_state_dim)
        else:
            layer = []
            if self._prop_type == 'embedding':
                layer.append(nn.Linear(self._node_state_dim * 3, self._node_hidden_sizes[0]))
            elif self._prop_type == 'matching':
                layer.append(nn.Linear(self._node_state_dim * 4, self._node_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
            self.MLP = nn.Sequential(*layer)

    def _compute_aggregated_messages(
            self, node_states, from_idx, to_idx, edge_features=None):
        """Compute aggregated messages for each node.

        Args:
          node_states: [n_nodes, input_node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.

        Returns:
          aggregated_messages: [n_nodes, aggregated_message_dim] float tensor, the
            aggregated messages for each node.
        """

        aggregated_messages = graph_prop_once(
            node_states,
            from_idx,
            to_idx,
            self._message_net,
            aggregation_module=None,
            edge_features=edge_features)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            reverse_aggregated_messages = graph_prop_once(
                node_states,
                to_idx,
                from_idx,
                self._reverse_message_net,
                aggregation_module=None,
                edge_features=edge_features)

            aggregated_messages += reverse_aggregated_messages

        if self._layer_norm:
            aggregated_messages = self.layer_norm1(aggregated_messages)

        return aggregated_messages

    def _compute_node_update(self,
                             node_states,
                             node_state_inputs,
                             node_features=None):
        """Compute node updates.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, the input node
            states.
          node_state_inputs: a list of tensors used to compute node updates.  Each
            element tensor should have shape [n_nodes, feat_dim], where feat_dim can
            be different.  These tensors will be concatenated along the feature
            dimension.
          node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.

        Returns:
          new_node_states: [n_nodes, node_state_dim] float tensor, the new node
            state tensor.

        Raises:
          ValueError: if node update type is not supported.
        """
        if self._node_update_type in ('mlp', 'residual'):
            node_state_inputs.append(node_states)
        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = torch.cat(node_state_inputs, dim=-1)

        if self._node_update_type == 'gru':
            node_state_inputs = torch.unsqueeze(node_state_inputs, 0)
            node_states = torch.unsqueeze(node_states, 0)
            _, new_node_states = self.GRU(node_state_inputs, node_states)
            new_node_states = torch.squeeze(new_node_states)
            return new_node_states
        else:
            mlp_output = self.MLP(node_state_inputs)
            if self._layer_norm:
                mlp_output = nn.self.layer_norm2(mlp_output)
            if self._node_update_type == 'mlp':
                return mlp_output
            elif self._node_update_type == 'residual':
                return node_states + mlp_output
            else:
                raise ValueError('Unknown node update type %s' % self._node_update_type)

    def forward(self,
                node_states,
                from_idx,
                to_idx,
                edge_features=None,
                node_features=None):
        """Run one propagation step.

        Args:
          node_states: [n_nodes, input_node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.
          node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.
        """
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features)

        return self._compute_node_update(node_states,
                                         [aggregated_messages],
                                         node_features=node_features)


class GraphAggregator(nn.Module):
    """This module computes graph representations by aggregating from parts."""

    def __init__(self,
                 node_hidden_sizes,
                 graph_transform_sizes=None,
                 input_size=None,
                 gated=True,
                 aggregation_type='sum',
                 name='graph-aggregator'):
        """Constructor.

        Args:
          node_hidden_sizes: the hidden layer sizes of the node transformation nets.
            The last element is the size of the aggregated graph representation.

          graph_transform_sizes: sizes of the transformation layers on top of the
            graph representations.  The last element of this list is the final
            dimensionality of the output graph representations.

          gated: set to True to do gated aggregation, False not to.

          aggregation_type: one of {sum, max, mean, sqrt_n}.
          name: name of this module.
        """
        super(GraphAggregator, self).__init__()

        self._node_hidden_sizes = node_hidden_sizes
        self._graph_transform_sizes = graph_transform_sizes
        self._graph_state_dim = node_hidden_sizes[-1]
        self._input_size = input_size
        #  The last element is the size of the aggregated graph representation.
        self._gated = gated
        self._aggregation_type = aggregation_type
        self._aggregation_op = None
        self.MLP1, self.MLP2 = self.build_model()

    def build_model(self):
        node_hidden_sizes = self._node_hidden_sizes
        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        layer = []
        layer.append(nn.Linear(self._input_size[0], node_hidden_sizes[0]))
        for i in range(1, len(node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
        MLP1 = nn.Sequential(*layer)

        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            layer = []
            layer.append(nn.Linear(self._graph_state_dim, self._graph_transform_sizes[0]))
            for i in range(1, len(self._graph_transform_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
            MLP2 = nn.Sequential(*layer)

        return MLP1, MLP2

    def forward(self, node_states, graph_idx, n_graphs):
        """Compute aggregated graph representations.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states of a
            batch of graphs concatenated together along the first dimension.
          graph_idx: [n_nodes] int tensor, graph ID for each node.
          n_graphs: integer, number of graphs in this batch.

        Returns:
          graph_states: [n_graphs, graph_state_dim] float tensor, graph
            representations, one row for each graph.
        """

        node_states_g = self.MLP1(node_states)

        if self._gated:
            gates = torch.sigmoid(node_states_g[:, :self._graph_state_dim])
            node_states_g = node_states_g[:, self._graph_state_dim:] * gates

        graph_states = unsorted_segment_sum(node_states_g, graph_idx, n_graphs)

        if self._aggregation_type == 'max':
            # reset everything that's smaller than -1e5 to 0.
            graph_states *= torch.FloatTensor(graph_states > -1e5)
        # transform the reduced graph states further


        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            graph_states = self.MLP2(graph_states)

        return graph_states

class GraphEmbeddingNet(nn.Module):
    """A graph to embedding mapping network."""

    def __init__(self,
                 encoder,
                 aggregator,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 n_prop_layers,
                 share_prop_params=False,
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 layer_class=GraphPropLayer,
                 prop_type='embedding',
                 name='graph-embedding-net'):
        """Constructor.

        Args:
          encoder: GraphEncoder, encoder that maps features to embeddings.
          aggregator: GraphAggregator, aggregator that produces graph
            representations.

          node_state_dim: dimensionality of node states.
          edge_hidden_sizes: sizes of the hidden layers of the edge message nets.
          node_hidden_sizes: sizes of the hidden layers of the node update nets.

          n_prop_layers: number of graph propagation layers.

          share_prop_params: set to True to share propagation parameters across all
            graph propagation layers, False not to.
          edge_net_init_scale: scale of initialization for the edge message nets.
          node_update_type: type of node updates, one of {mlp, gru, residual}.
          use_reverse_direction: set to True to also propagate messages in the
            reverse direction.
          reverse_dir_param_different: set to True to have the messages computed
            using a different set of parameters than for the forward direction.

          layer_norm: set to True to use layer normalization in a few places.
          name: name of this module.
        """
        super(GraphEmbeddingNet, self).__init__()

        self._encoder = encoder
        self._aggregator = aggregator
        self._node_state_dim = node_state_dim
        self._edge_state_dim = edge_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes
        self._node_hidden_sizes = node_hidden_sizes
        self._n_prop_layers = n_prop_layers
        self._share_prop_params = share_prop_params
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type
        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different
        self._layer_norm = layer_norm
        self._prop_layers = []
        self._prop_layers = nn.ModuleList()
        self._layer_class = layer_class
        self._prop_type = prop_type
        self.build_model()

    def _build_layer(self, layer_id):
        """Build one layer in the network."""
        return self._layer_class(
            self._node_state_dim,
            self._edge_state_dim,
            self._edge_hidden_sizes,
            self._node_hidden_sizes,
            edge_net_init_scale=self._edge_net_init_scale,
            node_update_type=self._node_update_type,
            use_reverse_direction=self._use_reverse_direction,
            reverse_dir_param_different=self._reverse_dir_param_different,
            layer_norm=self._layer_norm,
            prop_type=self._prop_type)
        # name='graph-prop-%d' % layer_id)

    def _apply_layer(self,
                     layer,
                     node_states,
                     from_idx,
                     to_idx,
                     graph_idx,
                     n_graphs,
                     edge_features):
        """Apply one layer on the given inputs."""
        del graph_idx, n_graphs
        return layer(node_states, from_idx, to_idx, edge_features=edge_features)

    def build_model(self):
        if len(self._prop_layers) < self._n_prop_layers:
            # build the layers
            for i in range(self._n_prop_layers):
                if i == 0 or not self._share_prop_params:
                    layer = self._build_layer(i)
                else:
                    layer = self._prop_layers[0]
                self._prop_layers.append(layer)

    def forward(self,
                node_features,
                edge_features,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs):
        """Compute graph representations.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: [n_edges, edge_feat_dim] float tensor.
          from_idx: [n_edges] int tensor, index of the from node for each edge.
          to_idx: [n_edges] int tensor, index of the to node for each edge.
          graph_idx: [n_nodes] int tensor, graph id for each node.
          n_graphs: int, number of graphs in the batch.

        Returns:
          graph_representations: [n_graphs, graph_representation_dim] float tensor,
            graph representations.
        """

        node_features, edge_features = self._encoder(node_features, edge_features)
        node_states = node_features

        layer_outputs = [node_states]

        for layer in self._prop_layers:
            # node_features could be wired in here as well, leaving it out for now as
            # it is already in the inputs
            node_states = self._apply_layer(
                layer,
                node_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                edge_features)
            layer_outputs.append(node_states)

        # these tensors may be used e.g. for visualization
        self._layer_outputs = layer_outputs
        return self._aggregator(node_states, graph_idx, n_graphs)

    def reset_n_prop_layers(self, n_prop_layers):
        """Set n_prop_layers to the provided new value.

        This allows us to train with certain number of propagation layers and
        evaluate with a different number of propagation layers.

        This only works if n_prop_layers is smaller than the number used for
        training, or when share_prop_params is set to True, in which case this can
        be arbitrarily large.

        Args:
          n_prop_layers: the new number of propagation layers to set.
        """
        self._n_prop_layers = n_prop_layers

    @property
    def n_prop_layers(self):
        return self._n_prop_layers

    def get_layer_outputs(self):
        """Get the outputs at each layer."""
        if hasattr(self, '_layer_outputs'):
            return self._layer_outputs
        else:
            raise ValueError('No layer outputs available.')
