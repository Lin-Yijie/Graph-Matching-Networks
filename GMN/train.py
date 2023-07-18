from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_loss
from utils import *
from configure import *
import numpy as np
import torch.nn as nn
import collections
import time
import os

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Print configure
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))

# Set random seeds
seed = config['seed']
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


training_set, validation_set = build_datasets(config)

if config['training']['mode'] == 'pair':
    training_data_iter = training_set.pairs(config['training']['batch_size'])
    first_batch_graphs, _ = next(training_data_iter)
else:
    training_data_iter = training_set.triplets(config['training']['batch_size'])
    first_batch_graphs = next(training_data_iter)

node_feature_dim = first_batch_graphs.node_features.shape[-1]
edge_feature_dim = first_batch_graphs.edge_features.shape[-1]

model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

accumulated_metrics = collections.defaultdict(list)


training_n_graphs_in_batch = config['training']['batch_size']
if config['training']['mode'] == 'pair':
    training_n_graphs_in_batch *= 2
elif config['training']['mode'] == 'triplet':
    training_n_graphs_in_batch *= 4
else:
    raise ValueError('Unknown training mode: %s' % config['training']['mode'])

t_start = time.time()
for i_iter in range(config['training']['n_training_steps']):
    model.train(mode=True)
    batch = next(training_data_iter)
    if config['training']['mode'] == 'pair':
        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
        labels = labels.to(device)
    else:
        node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(batch)
    graph_vectors = model(node_features.to(device), edge_features.to(device), from_idx.to(device), to_idx.to(device),
                          graph_idx.to(device), training_n_graphs_in_batch)

    if config['training']['mode'] == 'pair':
        x, y = reshape_and_split_tensor(graph_vectors, 2)
        loss = pairwise_loss(x, y, labels,
                             loss_type=config['training']['loss'],
                             margin=config['training']['margin'])

        is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
        is_neg = 1 - is_pos
        n_pos = torch.sum(is_pos)
        n_neg = torch.sum(is_neg)
        sim = compute_similarity(config, x, y)
        sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8)
        sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8)
    else:
        x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
        loss = triplet_loss(x_1, y, x_2, z,
                            loss_type=config['training']['loss'],
                            margin=config['training']['margin'])

        sim_pos = torch.mean(compute_similarity(config, x_1, y))
        sim_neg = torch.mean(compute_similarity(config, x_2, z))

    graph_vec_scale = torch.mean(graph_vectors ** 2)
    if config['training']['graph_vec_regularizer_weight'] > 0:
        loss += (config['training']['graph_vec_regularizer_weight'] *
                 0.5 * graph_vec_scale)

    optimizer.zero_grad()
    loss.backward(torch.ones_like(loss))  #
    nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_value'])
    optimizer.step()

    sim_diff = sim_pos - sim_neg
    accumulated_metrics['loss'].append(loss)
    accumulated_metrics['sim_pos'].append(sim_pos)
    accumulated_metrics['sim_neg'].append(sim_neg)
    accumulated_metrics['sim_diff'].append(sim_diff)


    # evaluation
    if (i_iter + 1) % config['training']['print_after'] == 0:
        metrics_to_print = {
            k: torch.mean(v[0]) for k, v in accumulated_metrics.items()}
        info_str = ', '.join(
            ['%s %.4f' % (k, v) for k, v in metrics_to_print.items()])
        # reset the metrics
        accumulated_metrics = collections.defaultdict(list)

        if ((i_iter + 1) // config['training']['print_after'] %
                config['training']['eval_after'] == 0):
            model.eval()
            with torch.no_grad():
                accumulated_pair_auc = []
                for batch in validation_set.pairs(config['evaluation']['batch_size']):
                    node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
                    labels = labels.to(device)
                    eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                       to_idx.to(device),
                                       graph_idx.to(device), config['evaluation']['batch_size'] * 2)

                    x, y = reshape_and_split_tensor(eval_pairs, 2)
                    similarity = compute_similarity(config, x, y)
                    pair_auc = auc(similarity, labels)
                    accumulated_pair_auc.append(pair_auc)

                accumulated_triplet_acc = []
                for batch in validation_set.triplets(config['evaluation']['batch_size']):
                    node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(batch)
                    eval_triplets = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                          to_idx.to(device),
                                          graph_idx.to(device),
                                          config['evaluation']['batch_size'] * 4)
                    x_1, y, x_2, z = reshape_and_split_tensor(eval_triplets, 4)
                    sim_1 = compute_similarity(config, x_1, y)
                    sim_2 = compute_similarity(config, x_2, z)
                    triplet_acc = torch.mean((sim_1 > sim_2).float())
                    accumulated_triplet_acc.append(triplet_acc.cpu().numpy())

                eval_metrics = {
                    'pair_auc': np.mean(accumulated_pair_auc),
                    'triplet_acc': np.mean(accumulated_triplet_acc)}
                info_str += ', ' + ', '.join(
                    ['%s %.4f' % ('val/' + k, v) for k, v in eval_metrics.items()])
            model.train()
        print('iter %d, %s, time %.2fs' % (
            i_iter + 1, info_str, time.time() - t_start))
        t_start = time.time()
