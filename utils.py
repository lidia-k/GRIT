import cProfile, pdb
import contextlib
import io
import os
import pstats
from copy import deepcopy
from datetime import datetime
from math import ceil

import dgl
import numpy as np
import pandas as pd
import torch
import torch.optim as opt
from dgl import DGLGraph
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Union

from data import samplers
from data.DatabaseDataset import DatabaseDataset
from data.TabularDataset import TabularDataset
from data.utils import get_db_info, train_val_split, five_fold_split_iter, get_ds_info

def setup_writer(log_dir, debug_network):
    if 'debug' in log_dir:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S-%f')
        log_dir = os.path.join('/tmp/RDB', log_dir, current_time)
    else:
        log_dir = os.path.join('runs', log_dir)

    if os.path.exists(log_dir) and 'debug' not in log_dir:
        raise FileExistsError(f'Manually delete {log_dir} if you want to overwrite its contents')

    writer = SummaryWriter(log_dir=log_dir)  #sn crashing here. torch.utils.tensorboard.SummaryWriter
    writer.verbose = True

    writer.batches_done = 0
    if debug_network:
        writer.debug_network = True
        writer.debug_histogram = writer.add_histogram
    else:
        writer.debug_network = False
        writer.debug_histogram = lambda *args: None
        writer.add_histogram = lambda *args: None

    return writer


def format_hparam_dict_for_tb(d, super_key=''):
    """
    tensorboard hparam logger needs a dict with all values as ints, floats, or strings.
    This function flattens a dict into that format.
    """
    hparam_dict = {}
    for k, v in d.items():
        k = super_key + '_' + k
        if isinstance(v, dict):
            hparam_dict.update(format_hparam_dict_for_tb(v, super_key=k))
        elif not isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = str(v)
        else:
            hparam_dict[k] = v
    return hparam_dict


def log_param_values(writer, model):
    for name, param in model.named_parameters():
        writer.add_histogram('Parameter Values/{}'.format(name), param, writer.batches_done)
        if param.requires_grad and param.grad is not None:
            writer.add_histogram('Gradients/{}'.format(name), param.grad, writer.batches_done)


def truncate_graph(db_info, max_nodes_per_graph, edge_list, node_types, edge_types, features):
    """
    Returns a trunated copy of edge_list, node_types, edge_types, and features.
    Removes all nodes with index >= max_nodes_per_graph
    """
    # Cutoff nodes
    cutoff_node_types = node_types[:max_nodes_per_graph]
    # Cutoff edges
    cutoff_edge_list = []
    cutoff_edge_types = []
    for (u, v), type in zip(edge_list, edge_types):
        if u < max_nodes_per_graph and v < max_nodes_per_graph:
            cutoff_edge_list.append((u, v))
            cutoff_edge_types.append(type)
    # Cutoff features
    cutoff_features = {}
    for node_type, g_features in features.items():
        n_nodes_this_type = cutoff_node_types.count(db_info['node_type_to_int'][node_type])
        cutoff_features[node_type] = {}
        for feature_name, feature_values in g_features.items():
            cutoff_feat_values = features[node_type][feature_name][:n_nodes_this_type]
            cutoff_features[node_type][feature_name] = cutoff_feat_values

    return cutoff_edge_list, cutoff_node_types, cutoff_edge_types, cutoff_features


def nan_initializer(shape, dtype, ctx, id_range):
    init = torch.empty(shape, dtype=dtype, device=ctx)
    init[:] = np.nan
    return init

# intent: Concatenate a batch of datapoints together
# output: b_dgl, b_features
# input : datapoints (d) = a batch of subgraphs
#          d[0][0]     d[0][1]                                 
#               |      |    d[0][1][3]['Essay']['titleV'][0] =   |----------|     = string
#               |      |    d[0][1][3]['Essay']['titleV']    = |--------------|   = 1-element array
#               |      |    d[0][1][3]['Essay']     = |-----------------------------------|
#               |      |    d[0][1][3]  =  |-------------------------------------------------------------|
#  datapoints   V      V
#  = [ 0:  ( 'ba..8b', ( edges, nodes, xx, { p:{..} e:{ titleV:[ '[2.8, 4.6]' ], essayV:..} r:{..} rt:{..} } )
#      1:  ( '3a..f8', ...
#      2:  ( '0f..57', ...  ]
#
#d /250g/home/snguyen/.local/lib/python3.10/site-packages/dgl/heterograph.py:92: 
#   DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.

def get_DGL_collator(feature_encoders, db_info, max_nodes_per_graph=False):
  def DGL_collator(datapoints):     # t = time.perf_counter()
    dgl_graphs = [];   b_node_types = []; b_dp_ids = []; 
    b_features = None; b_edge_types = []; labels   = [];
        
    for dp_id, (edge_list, node_types, edge_types, features, label) in datapoints:
      b_dp_ids.append(dp_id)                                 # print(dp_id, len(node_types), len(edge_types)) 
      # Truncate enormous graphs if necessary
      if max_nodes_per_graph and len(node_types) > max_nodes_per_graph:         # print(f'Cutting off graph {dp_id}')
          edge_list, node_types, edge_types, features = truncate_graph( db_info
          ,   max_nodes_per_graph, edge_list, node_types, edge_types, features)
      edge_list  += [(v, u) for u, v in edge_list ]          # Add reverse edges
      edge_types += [-i     for i    in edge_types]          # Add reverse edges
      edge_list += [(i, i) for i in range(len(node_types))]  # Add self-edges
      edge_types += [0] * len(node_types)                    # Add self-edges
      sor = [ i[0] for i in edge_list ]                      #sn added
      des = [ i[1] for i in edge_list ]                      #sn added
      graph = dgl.graph( data = (sor,des) ).to('cuda')   #d  #sn was:   graph = DGLGraph(graph_data=edge_list)
      dgl_graphs.append(graph)
      b_node_types.append(node_types)
      b_edge_types.append(edge_types)
      features = RemoveEssayText( features ) #sn
      if b_features is None:
        b_features = deepcopy(features)
      else:
        for node_type, g_features in features.items():
          for feature_name, feature_values in g_features.items():
            b_features[node_type][feature_name] += feature_values
      b_features = RemoveEssayText( b_features )  #sn Removethis is probably redundant.
      labels.append(label)

    b_dgl = dgl.batch(dgl_graphs)
    b_dgl.set_n_initializer(nan_initializer)
    b_dgl.set_e_initializer(nan_initializer)
    b_node_types = torch.LongTensor(np.concatenate(b_node_types))
    b_edge_types = torch.LongTensor(np.concatenate(b_edge_types))
    b_dgl.dp_ids = b_dp_ids
    b_dgl.ndata['node_types'] = b_node_types.to('cuda')  #sn added to(cuda)
    b_dgl.edata['edge_types'] = b_edge_types.to('cuda')  #sn added to(cuda)
    # print('build DGLGraphs: {}'.format(time.perf_counter() - t))
    # Encode the batch features into Tensors from their database values
    # t = time.perf_counter()
    missing_node_types = []

    #a special treatment for essay.  we should really try to create a separate 
    #  DoNothing encoder, but it caused errors.
    #b example:  feature_values =  list of strings = ['[1,..4]','[2,..5]'].  
    #    s='[1,4]' --> torch.from_numpy( np.array( s.replace('[','').replace(']','').split(',') ).astype(float) )
    #    xx = tuple( [ torch.from_numpy( np.array( s.replace('[','').replace(']','').split(',') ).astype(float) ) for s in feature_values ] )
    #e list of 2 arrays -->  tensor 2x384.  batchZ = 2
    #c for Essay:  [t t t t] each t is 2x384 --> 2x1536.
    #  dim=1 is horizontal gluing.  dim=0 is vertical.
    #
    # 3 nested loops:  projects (datapoints) --> node_type (P,E,R,RT) --> column (features)
    ii = 0
    for node_type, features in b_features.items():  # items() = project:{..}, essay:{..}, resource:{..}
      cat_features = []
      cont_features = []
      for feature_name, feature_values in features.items():  # items() = { tV:[..], eV:[..], ..},  {..}
          encoder = feature_encoders[node_type][feature_name]
          if not feature_values:  # In case there are no nodes of this type in the batch
            assert all(f == [] for f in features.values())
            missing_node_types.append(node_type)
            break
          else:
            feature_values = pd.Series(feature_values)
            cat_feats      = encoder.enc_cat( feature_values)
            if cat_feats   is not None:   cat_features.append(cat_feats)
#            if feature_name == 'project_id':  print( f'project = {feature_values}' )  #sn
#            print( f'{feature_name}:  {feature_values}' ) #sn
#            ii += 1; print( f'ii = {ii} feature_name = {feature_name}' ) #sn
            cont_feats     = encoder.enc_cont(feature_values) 
            '''-----------  if bfloat16 errors, put array on cuda or do tensor.bfloat16()
            if node_type == 'Essay' and feature_name in ['essayV','titleV', 'need_statementV'
            ,   'short_descriptionV']:
                if isinstance( cont_feats, torch.Tensor ):
                    cont_feats = cont_feats.bfloat16()
                    print( f'grit/utils.py DGL_collator() - cont_feats = {cont_feats}' )
                    import pdb;  pdb.set_trace() 
            --------------'''
            if cont_feats  is not None:  cont_features.append(cont_feats) 
      if cat_features:     cat_data  = torch.cat( cat_features, dim=1) 
      else:                cat_data  = []
      if cont_features:    cont_data = torch.cat(cont_features, dim=1) #c
      else:                cont_data = []
      b_features[node_type] = (cat_data, cont_data)
    for nt in missing_node_types:  del b_features[nt]
    try:              b_label = torch.LongTensor(labels)        # Collate label
    except TypeError: b_label = None  # This batch is from the test set
    # print('encode features and label: {}'.format(time.perf_counter() - t))
    return (b_dgl, b_features), b_label
  return DGL_collator

def RemoveEssayText( features ):  #sn new function
   names = list( features['Essay'].keys() )
   for feature_name in names:
       if feature_name in ['essay','title','need_statement','short_description']:
           features['Essay'].pop( feature_name )
   return features

def get_train_test_dp_ids(dataset_name):
    db_name = None
    if 'acquirevaluedshopperschallenge' in dataset_name:
        db_name = 'acquirevaluedshopperschallenge'
    elif 'homecreditdefaultrisk' in dataset_name:
        db_name = 'homecreditdefaultrisk'
    elif 'kddcup2014' in dataset_name:
        db_name = 'kddcup2014'
    if db_name is not None:
        db_info = get_db_info( db_name, keeptext=False )  #sn added keeptext argument
        train_dp_ids = db_info['train_dp_ids']
        test_dp_ids = db_info['test_dp_ids']
    else:
        ds_info = get_ds_info(dataset_name)
        n_datapoints = ds_info['meta']['n_datapoints']
        train_dp_ids = np.arange(n_datapoints)
        test_dp_ids = None
    return train_dp_ids, test_dp_ids


def get_train_val_test_datasets(dataset_name, train_test_split, encoders, train_fraction_to_use=1.0):
    assert train_test_split in ['use_full_train', 'xval0', 'xval1', 'xval2', 'xval3', 'xval4']
    train_dp_ids, test_dp_ids = get_train_test_dp_ids(dataset_name)

    if train_test_split == 'use_full_train':
        train_dp_ids, val_dp_ids = train_val_split(train_dp_ids)
        test_dp_ids = np.array(test_dp_ids)
    else:
        fold = int(train_test_split[-1])
        trainval_dp_ids, test_dp_ids = list(five_fold_split_iter(train_dp_ids))[fold]
        train_dp_ids, val_dp_ids = train_val_split(trainval_dp_ids)
    if train_fraction_to_use != 1.0:
        assert 0.0 < train_fraction_to_use < 1.0
        train_dp_ids = train_dp_ids[:ceil(train_fraction_to_use * len(train_dp_ids))]

    if dataset_name in ['acquirevaluedshopperschallenge', 'homecreditdefaultrisk', 'kddcup2014']:
        # Load up graph datasets
        train_dataset = DatabaseDataset(dataset_name, train_dp_ids, encoders)
        val_dataset   = DatabaseDataset(dataset_name, val_dp_ids  , encoders)
        test_dataset  = DatabaseDataset(dataset_name, test_dp_ids , encoders)
    else:
        # Load up tabular datasets
        train_dataset = TabularDataset(dataset_name, train_dp_ids, encoders)
        train_dataset.fit_feat_encoders()
        # Handling mismatch between feature encoders when not using full training set
        if train_fraction_to_use == 1.0:
            fe = train_dataset.feature_encoders
        else:
            temp_train_ds, _, _ = get_train_val_test_datasets(dataset_name,
                                                              train_test_split,
                                                              encoders,
                                                              train_fraction_to_use=1.0)
            fe = temp_train_ds.feature_encoders
        train_dataset.encode(fe)
        val_dataset = TabularDataset(dataset_name, val_dp_ids, encoders)
        val_dataset.encode(fe)
        test_dataset = TabularDataset(dataset_name, test_dp_ids, encoders)
        test_dataset.encode(fe)
    trueK = 0; falseK = 0; total = len(val_dataset) #sn
    for i in range( total ):  #sn
#      print( f'is_exciting = {train_data[i][1][4]}' )
      if val_dataset[i][1][4] == True :  trueK =  1 + trueK  #sn
      if val_dataset[i][1][4] == False: falseK =  1 + falseK  #sn
    print( f'total = {total}.  trueK = {trueK/total}.  falseK = {falseK/total}' ) #sn

    return train_dataset, val_dataset, test_dataset


def get_optim_with_correct_wd(optimizer_class_name, model, optimizer_kwargs,
                              wd_bias=False, wd_embed=False, wd_bn=False):
    # In general, it may not be good to have weight_decay on bias terms, embeddings, or batch norm parameters
    if 'weight_decay' in optimizer_kwargs:
        no_wd_params = []
        wd_params = []
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Embedding) and not wd_embed:
                no_wd_params.append(m.weight)
            elif isinstance(m, torch.nn.BatchNorm1d):
                if wd_bn:
                    wd_params += m.parameters()
                else:
                    no_wd_params += m.parameters()
            else:
                for np, p in m.named_parameters(recurse=False):
                    if 'bias' in np and not wd_bias:
                        no_wd_params.append(p)
                    else:
                        wd_params.append(p)
        assert len(wd_params) + len(no_wd_params) == len(list(model.parameters()))
        params = [{'params': no_wd_params, 'weight_decay': 0.0},
                  {'params': wd_params}]
    else:
        params = model.parameters()
    return opt.__dict__[optimizer_class_name](params, **optimizer_kwargs)


def model_to_device(model, device_id):
    if torch.cuda.is_available() and 'cuda' in device_id:
        try:
            device_id = int(device_id[-1])
            torch.cuda.set_device(device_id)
        except ValueError:
            pass
        model.cuda()
        model.device = torch.device(torch.cuda.current_device())
    else:
        print(f'Falling back to CPU from requested device {device_id}')
        model.device = torch.device('cpu')


def get_dataloader(dataset: Union[DatabaseDataset, TabularDataset],
                   batch_size,
                   sampler_class_name=None,
                   sampler_class_kwargs={},
                   num_workers=0,
                   max_nodes_per_graph=False):
    batch_sampler = None
    sampler = None
    collate_fn = None
    sampler = samplers.__dict__[sampler_class_name](dataset, **sampler_class_kwargs)

    if isinstance(dataset, DatabaseDataset):
        collate_fn = get_DGL_collator(dataset.feature_encoders,
                                      dataset.db_info,
                                      max_nodes_per_graph=max_nodes_per_graph)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        batch_sampler=batch_sampler,
                        num_workers=num_workers,
                        collate_fn=collate_fn,
                        pin_memory=True)
    return loader


class DummyWriter:
    verbose = False

    def __init__(self):
        self.batches_done = 0

    def add_histogram(self, *args, **kwargs):
        pass

    def add_scalar(self, tag_name, object, iter_number, *args, **kwargs):
        if tag_name == 'Train Loss/Train Loss':
            # For testing purposes
            self.train_loss = object
            print('Train Loss step {} = {}'.format(iter_number, object))
        else:
            pass

    def debug_info(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass


@contextlib.contextmanager
def profiled():
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    # uncomment this to see who's calling what
    # ps.print_callers()
    print(s.getvalue())
