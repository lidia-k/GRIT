#------------------------------------------------------------------------------
# intent: vectorize essay text stored in neo4j kddcup2014 database
# input : kdd graph database text fields:  essay, title, short_description
#         need_statement
# output: vectors essayV, titleV, short_descriptionV, need_statementV
# method:
# - grab all essays, project_id from neo4j
# - batch and vectorize.
# - write vector to neo4j.  
# - re-create pickle files w/ vector inside it.
#------------------------------------------------------------------------------

# suffixes:  Z = size, K = count, V=vector, B = Batch
# create (j:junk {vector: [1.1, 2.2]})  return j;   # create
# match  (n:junk) set n.vectorA = [6,7] return n;   # update or add column
# match  (j:junk) delete j;                         # delete
# match  ( n:junk {vector:[1.1,2.2]} )              # update where
# . set  n.vectorA = [8,9]              return n;

# match  (e:Essay) -[r:ESSAY_TO_PROJECT]-> (p:Project) 
# where  e.titleV is null and e.essayV is not null 
# return count(e)
#
# use vectors as floats
# create ( n:planet { name:'earth', size:toFloatList( ['2','3','4'] ) } );

import os, time, neo4j, numpy, re 
from   data.utils import get_neo4j_db_driver
from   data.data_encoders import TextEmbeddingsEnc


vectorizer = TextEmbeddingsEnc()
batchZ  = 32 # Z = size
j = 0  

db_name = 'kddcup2014'
driver = get_neo4j_db_driver(db_name)

query   = ''.join([ 'match  (e:Essay) -[r:ESSAY_TO_PROJECT]-> (p:Project) '
,         'where e.essayV is null return e, r, p limit 100000' ])
#,        'where e.short_descriptionV is null and e.essayV is not null return e, r, p limit 3' ])

#b min(): at the last batch, we want to end at len(pan), not start + batchZ
#c eg:  match x = (e:Essay)-[r:ESSAY_TO_PROJECT]->(p:Project 
#       {    project_id: 'ffffc4f85b60efc5b52347df489d0238' } ) 
#       set  e.essayV = '[1,2,3]' 
#d "r" = raw string, so python doesn't treat "\[" as an escaped unicode character

print( "-------------- start time ------------------  " ); os.system('date')

with driver.session() as session:
  pan = driver.execute_query(
    query, database='kddcup2014', result_transformer_ = neo4j.Result.to_df
  )

  for feature in ['essay','title','need_statement','short_description']:
    for start in range( 0, len(pan), batchZ ):    # loop batches.  start = 0, 32, 64, ...
      end        = min( start + batchZ, len(pan) )                           #b
      
      text_batch = [ pan['e'][i][feature] for i in range( start, end ) ]
      projects   = [ pan['p'][i]['project_id'] for i in range( start, end ) ]
      vectors    = vectorizer.embed( text_batch )

      start_time = time.time();               # perf timer
      duration   = start_time - time.time()
      if end % (batchZ * 10) == 0: 
        print(f'{feature}.  {end}.  {duration}')
      
      for i in range( len(vectors) ):
        vectorString = numpy.array2string( vectors[i] )
        vectorString = re.sub( "\s+"  , ",", vectorString.strip() )
        vectorString = re.sub( r'^\[,', '[', vectorString.strip() )          #d
        query = ''.join([ "match x = (e:Essay) -[r:ESSAY_TO_PROJECT]-> "
        ,  "(p:Project { project_id: \'", projects[i], "\' } ) "
        ,  "set e.",feature,"V = \'", vectorString, "\'" ])                  #c
        update = driver.execute_query( query, database='kddcup2014' )

print( "end time: " ); os.system('date')
