#-------------------------------------------------------------------------------
# intent: create pickle subgraph files for training.
#         ideally RDBtoGraph would be implemented, but Milan says he did not.
# input : json file, neo4j data
# output: 600+k files in preprocessed_datapoints directory. 1 file for 1 project.
#         a file contains: edge_list, node_types, edge_types, features, label
# usage : python3 -m data.kddcup2014.build_dataset_from_database-vector
#         run from the "grit" directory
# assume:
# run kddcup2014.db_info.json first.
# dbinfo.json defines the features (columns) for each node.  this script uses 
# dbinfo.json to list the features to grab and dump into the pickle files; 
# therefore <essay>V features must be in there.  since the <essay>V features 
# were added in the database, dbinfo.json must be recreated before we run this 
# script.  
#
# method: 
# 1. read file kddcup2014.db_info.json into db_info.  these are attributes
#  . this file is supplied in github, but is also written to in step 8. 
#  . odd circular logic.  step 8 should be step 6.
# 2. get list of projects (neo4j --> datapoint_ids)
# 3. loop pids: get subgraph starting from a project node write tuple
#  .            (edge_list, node_types, edge_types, features, label)
#  .            to file preprocessed_datapoints/<pid>. 
# build_dataset_from_database.py
# --> create_datapoints_with_xargs( ... match...where...return... )
#     --> Popen( cat {} | xargs ... create_datapoint_from_database ... )
#         --> run( query )
#
# sample data structure:  5 nodes, 7 edges.  dumped into output files
# . datapoint_ids[i]: '316ed8fb3b81402ff6ac8f721bb31192' (project_id)
# . edge_list       : [(1, 0), (1, 2), (3, 0), (3, 2), (4, 0), (4, 2), (0, 2)]
# . node_types      : [0, 2, 3, 2, 2]
# . edge_types      : [3, 4, 3, 4, 3, 4, 1]
# . features        :
# .  { 'Project': {'school_id': ['c0e6ce...'], 'school_county': ['Fresno']
# ,       (~30 more columns) ... 'total_price...': [555.81] }
# ,    'Essay': {'title':[], 'short_description':[], 'need_statement':[], 'essay':[]}
# ,    'Resource': {'vendor_id': ['7', '27', '7']
# ,      'vendor_name': ['AKJ Books', 'Amazon', 'AKJ Books']
# ,      'item_name': ['Number the Stars', 'Seedfolks', 'Esperanza Rising']
# ,      'item_number': ['9780547577098', '0064472078', '9780439120425']
# ,      'item_unit_price': [5.1, 4.2, 5.1], 'item_quantity': [33, 33, 33]}
# ,    'ResourceType': {'resource_type': ['Books']} }
#
# notes :
# merged 3 files into this one: 
#   (1) this file - build_dataset_from_database.py
#   (2) data/utils.py::create_datapoints_with_xargs() 
#   (3) data/create_datapoint_from_database.py
# sn rewrite:  opens up create_datapoint_from_database() as main()
#   so that we don't have to do xargs shell command.
#-------------------------------------------------------------------------------

import os, pdb, pickle, sys, neotime, neo4j #sn added neo4j for isinstance( value, (neo4j.time.Date) )
from __init__ import data_root
from data.utils import create_datapoints_with_xargs, get_neo4j_db_driver
from data.utils import get_db_info
from data.utils import get_neo4j_db_driver

#sn was:  if __name__ == '__main__':
#a https://neo4j.com/docs/cypher-cheat-sheet/5/auradb-enterprise/#_variable_length_relationships
#    Variable-length path of between 0 and 1 hop between two nodes.
db_name = 'kddcup2014'
driver = get_neo4j_db_driver(db_name)
with driver.session() as session:
  datapoint_ids = session.run('MATCH (p:Project) RETURN p.project_id').value()
  base_query = 'MATCH r = (p:Project)--(n)-[*0..1]->(m) \
    WHERE p.project_id = "{}"  RETURN p, r, n, m'                            #a
target_dir = os.path.join(data_root, db_name, 'preprocessed_datapoints')
# os.makedirs(target_dir, exist_ok=False)
n_jobs = 20
#sn was:  create_datapoints_with_xargs(db_name, datapoint_ids, base_query, target_dir, n_jobs)

#-----------------------------
# what follows below avoids create_datapoints_with_xargs() by pasting the body 
# of data/create_datapoint_from_database.py
#-----------------------------

db_info = get_db_info(db_name)
driver = get_neo4j_db_driver(db_name)
for i in range( len(datapoint_ids) ):                                 # loop graphs (project)
  # Get graph from database
  with driver.session() as session:
    query = base_query.format(datapoint_ids[i]); result = session.run(query)
    g = result.graph()

  # Construct DGLGraph for each neo4j graph, and batch them
  # Also collect the features and labels from each graph
  label_node_type, label_feature_name = db_info['label_feature'].split('.')
  features = {}
  for node_type in db_info['node_types_and_features'].keys():
    features[node_type] = {}
    for feature_name in db_info['node_types_and_features'][node_type].keys():
      # Making sure not to include the label value among the training features
      if not (node_type == label_node_type and feature_name == label_feature_name):
        features[node_type][feature_name] = []

  neo4j_id_to_graph_idx = {node.element_id: idx for idx
  ,   node in enumerate(g.nodes)}
  node_types = [None] * len(g.nodes)
  for node in g.nodes:                                                 # loop node
    node_type = tuple(node.labels)[0]
    node_idx = neo4j_id_to_graph_idx[node.element_id]
    node_types[node_idx] = db_info['node_type_to_int'][node_type]

    for feature_name, feature_values in features[node_type].items():   # loop feature
      # Dealing with latlongs
      if db_info['node_types_and_features'][node_type][feature_name]['type'] == 'LATLONG':
            lat_name, lon_name = feature_name.split('+++')
            value = (node.get(lat_name), node.get(lon_name))
      else: value = node.get(feature_name)
      # neotime doesn't pickle well
      if isinstance(value, (neo4j.time.Date, neotime.Date, neotime.DateTime)):  #sn added "neo4j.time.Date"
        value = value.to_native()
      feature_values.append(value)
    if node_type == label_node_type:  label = node.get(label_feature_name)

  edge_list = []; edge_types = []
  for rel in g.relationships:
    start_node_idx = neo4j_id_to_graph_idx[rel.start_node.element_id]
    end_node_idx = neo4j_id_to_graph_idx[rel.end_node.element_id]
    edge_list.append((start_node_idx, end_node_idx))
    edge_types.append(db_info['edge_type_to_int'][rel.type])

  with open(os.path.join(target_dir, str(datapoint_ids[i])), 'wb') as f:
    dp_tuple = (edge_list, node_types, edge_types, features, label)
    pickle.dump(dp_tuple, f)
