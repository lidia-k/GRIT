import json
import os
import time 

import pandas as pd

from data.data_encoders import TextEmbeddingsEnc
from data.utils import get_neo4j_db_driver


def rec_val_generator(generator):
    while True:
        try:
            n = generator.__next__()
            yield n if n else ''
        except StopIteration:
            return

file_path = os.path.join(os.getcwd(), 'data/kddcup2014/kddcup2014.db_info.json')
with open(file_path, 'r') as f:
    data = json.load(f)

db_name = 'kddcup2014'
driver = get_neo4j_db_driver(db_name)

with driver.session() as session:
    text_embedder = TextEmbeddingsEnc()
    essay_features = data['node_types_and_features']['Essay']

    for feature in essay_features:
        query = f'MATCH (n:Essay) RETURN n.{feature} ;'
        print(f'running: {query}')
        text_strings = session.run(query).value().__iter__()
        text_strings = pd.Series(rec_val_generator(text_strings))

        output = []
        batch_size = 32
        for start_idx in (0, len(text_strings), batch_size):
            end_idx = start_idx + batch_size 
            text_batch = text_strings[start_idx:end_idx]

            start_time = time.time()
            
            embeddings = text_embedder.embed(text_batch.tolist())
            embeddings_list = embeddings.tolist()
            
            end_time = time.time()
            print(f'{end_time - start_time}')

            output.extend(embeddings_list)

        essay_features[feature]['Text_embeddings_'] = output

with open(file_path, 'w') as f:
    json.dump(data, f, indent=1, allow_nan=False)