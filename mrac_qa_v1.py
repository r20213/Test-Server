'''
Installation Process:

1. ! wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
	 ! tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
	 ! chown -R daemon:daemon elasticsearch-7.9.2
   from subprocess import Popen, PIPE, STDOUT

2. Run Server:

server_file = 'elasticsearch-7.9.2/bin/elasticsearch'

es_server = Popen([server_file],
                   stdout=PIPE, stderr=STDOUT,
                   preexec_fn=lambda: os.setuid(1)  # as daemon
                  )
! sleep 30

3. !pip install -r requirements.txt
'''

'''
Instructions to Run:

1. Complete Installation process.
2. cd into the folder - Make sure all models are downloaded and files are available.
3. 
'''

# Install Libraries

try:
	import json
	from haystack.preprocessor.cleaning import clean_wiki_text
	from haystack import Finder
	from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
	from haystack.reader.farm import FARMReader
	from haystack.retriever.sparse import ElasticsearchRetriever
	from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
	from haystack.utils import print_answers
	from haystack.preprocessor import PreProcessor
	from haystack import Pipeline
	from haystack.pipeline import JoinDocuments
	from haystack.retriever.dense import DensePassageRetriever
	from subprocess import Popen, PIPE, STDOUT
except:
	import json
	from haystack.preprocessor.cleaning import clean_wiki_text
	from haystack import Finder
	from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
	from haystack.reader.farm import FARMReader
	from haystack.retriever.sparse import ElasticsearchRetriever
	from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
	from haystack.utils import print_answers
	from haystack.preprocessor import PreProcessor
	from haystack import Pipeline
	from haystack.pipeline import JoinDocuments
	from haystack.retriever.dense import DensePassageRetriever
	from subprocess import Popen, PIPE, STDOUT


'''
Check Installation of Elastisearch Server Before Running Below
'''

class MRAC_QA:

  '''
  Inputs for Query Method:
    1. Question - Correctly Formatted Text Question
    2. Number of Replies Needed (Per Retriever) - Default: 5*2 = 10
    3. If Context Needed for Answer - True/False - Default: False

  Check dir before running.
  '''

  def __init__(self, dir="", split_length=100, model = "ahotrod/albert_xxlargev1_squad2_512"):
    self.document_store = ElasticsearchDocumentStore(index="document")
    self.dir = dir
    self.split_length = split_length
    self.model = model
    self.pipeline = Pipeline()

    # Call Main
    self.main()


  '''
  Preprocessor for text.
  Only activated if preprocessed data is not available.
  '''
  def get_preprocessor(self):
    preprocessor = PreProcessor(
      clean_empty_lines=True,
      clean_whitespace=True,
      clean_header_footer=True,
      split_by="word",
      split_length=self.split_length,
      split_respect_sentence_boundary=True
    )
    return preprocessor


  '''
  Helper for preprocessor
  Only activated if preprocessed data is not available.
  '''
  def get_preprocessed_dict(self, dicts):
    preprocessor = self.get_preprocessor()
    nested_docs = [preprocessor.process(d) for d in dicts]
    docs = [d for x in nested_docs for d in x]
    return docs


  '''
  Create data.
  Only activated if data is not available.
  '''
  def get_data_dict(self):
    dicts = convert_files_to_dicts(dir_path='Data/NLP/', clean_func=clean_wiki_text, split_paragraphs=True)
    print('Number of Files -', len(dicts))
    return dicts

  
  '''
  Main Data Loader.
  '''
  def load_data(self):
    try:
      docs_dir = 'Data/nlp_v1.json'
      with open(docs_dir, 'r') as f:
        dicts = json.load(f)
      print('Found Data')
      self.document_store.write_documents(dicts)
      print('Documents Saved in Document Store')
    except:
      print('Could not find any presaved data') 
      dicts = self.get_data_dict()
      dicts = self.get_preprocessed_dict(dicts)
      self.document_store.write_documents(dicts)
      with open('Data/nlp_v1.json', 'w') as f:
        json.dump(dicts, f)
      print('Documents Saved in Document Store')


  '''
  Gets both retrievers - DPR and ES.
  Skips DPR if already available pretrained.
  '''
  def get_retriever(self, is_es_true):
    if (is_es_true is False):
      retriever = DensePassageRetriever(document_store=self.document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-multiset-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-multiset-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=256,
                                  batch_size=16,
                                  use_gpu=True,
                                  embed_title=True,
                                  use_fast_tokenizers=True)
      self.document_store.update_embeddings(retriever)
      return retriever
    else:
      retriever = ElasticsearchRetriever(document_store=self.document_store)
      return retriever


  '''
  Gets Reader - ALBERT.
  Skips if already available pretrained.
  '''
  def get_reader(self):
    reader = FARMReader(model_name_or_path=self.model, use_gpu=True, context_window_size=300, progress_bar=False)
    return reader


  '''
  Pipeline Builder.
  Sets up all components into Pipeline Object.

  Add to first try if you want to save a new retriever:
        path_to_save = "Models/dpr_retriever"
      	dpr.save(path_to_save)
  '''
  def build_pipeline(self):
    try:
      print('Found Retriever')
      query_model = "Models/dpr_retriever/query_encoder"
      passage_model = "Models/dpr_retriever/passage_encoder"
      dpr = DensePassageRetriever(document_store=self.document_store,
                                  query_embedding_model=query_model,
                                  passage_embedding_model=passage_model, use_gpu=True)
    except Exception as e:
      print(e)
      print('Could not find saved retriever. Loading DPR from source.')
      dpr = self.get_retriever(False)
    try:
      print('Found Reader')
      reader_model = 'NewModel'
      mrc = FARMReader(model_name_or_path=reader_model, progress_bar=False)
    except:
      print('Could not find saved reader. Loading Reader from source.')
      mrc = self.get_reader()

    es = self.get_retriever(True)

    self.pipeline.add_node(component=es, name="ElasticRetriever", inputs=["Query"])
    self.pipeline.add_node(component=dpr, name="DeepRetriever", inputs=["Query"])
    self.pipeline.add_node(component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["ElasticRetriever", "DeepRetriever"])
    self.pipeline.add_node(component=mrc, name="AlbertLargerReader", inputs=["JoinResults"])


  '''
  MAIN COMPONENT OF CODE
  '''
  def main(self):
    if (self.document_store.get_document_count() != 0):
        print('All files already here')
    else:
        print('LOADING DATA AND UPLOADING TO SERVER')
        self.load_data()
    print('DATABASE SETUP COMPLETE')
    print('BUILDING PIPELINE')
    pipe = self.build_pipeline()
    print('PIPELINE BUILD SUCCESSFULL. ASK A QUERY.')

  def query(self, question, num_results=5, numcontext=0):
    prediction = self.pipeline.run(query=question, top_k_retriever=10, top_k_reader=num_results)
    answer = []
    context = []
    j = 0

    for i in prediction['answers']:
      if (j == num_results):
        break
      print('Possible Answer - ', i['answer'])
      answer.append(i['answer'])

      if (numcontext == 0):
        j += 1
        continue

      con = self.document_store.get_document_by_id(i['document_id']).text
      si = i['offset_start_in_doc']
      ei = i['offset_end_in_doc']

      finalcon = con[:si] + "<mark style='background-color=yellow; color:black;'>" + con[si:ei] + "</mark>" + con[ei:]

      context.append(finalcon)

      j += 1

    return answer, context

  def discord_query(self, question, num_results=5, numcontext=0):
    prediction = self.pipeline.run(query=question, top_k_retriever=10, top_k_reader=num_results)
    answer = []
    context = []
    j = 0
    document = []

    for i in prediction['answers']:
      if (j == num_results):
        break
      print('Possible Answer - ', i['answer'])
      answer.append(i['answer'])
      document.append(i['meta']['name'])

      if (numcontext == 0):
        j += 1
        continue

      con = self.document_store.get_document_by_id(i['document_id']).text
      si = i['offset_start_in_doc']
      ei = i['offset_end_in_doc']

      finalcon = con[:si] + "**`" + con[si:ei] + "`**" + con[ei:]

      context.append(finalcon)

      j += 1

    return answer, context, document

  def doc_retrieve(self, query, num_passages=3):
    es = self.pipeline.get_node("DeepRetriever")
    docs = es.retrieve(query, top_k = num_passages)
    text = ''

    for i in docs:
      if (text == ''):
        text = text + i.text
      else:
        text = text + ' ' + i.text

    return text

