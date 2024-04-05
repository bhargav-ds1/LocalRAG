from DataHouse.Readers import DefaultIngest
from DataHouse.Readers import NotionAPIPageReader
from llama_index.core.storage.docstore import SimpleDocumentStore
import json
import time
import os


class GetDataFromNotionAPI(DefaultIngest):
    def __init__(self, notion_api_key: str, re_download: bool = True, recursive: bool = True, num_processes: int = 2,
                 verbose: bool = False,
                 output_dir: str = './DataHouse/Data/notion-ingest-output',
                 download_dir: str = './DataHouse/Data/Llama_index/rawData',
                 work_dir: str = './DataHouse/Data/workDir',
                 page_ids: list = None, database_ids: list = None,
                 get_embeddings: bool = False,
                 embedding_provider: str = 'langchain-huggingface',
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L12-v2',
                 embedding_api_token: str = None,
                 chunking_strategy: str = 'basic', chunk_multipage: bool = True, chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 store: str = 'simple', store_settings: dict = None, store_name: str = None
                 ):
        super().__init__()
        self.notion_api_key = notion_api_key
        self.re_download = re_download
        self.recursive = recursive
        self.num_processes = num_processes
        self.verbose = verbose
        self.output_dir = output_dir
        self.download_dir = download_dir
        self.work_dir = work_dir
        self.page_ids = page_ids or []
        self.database_ids = database_ids or []
        self.get_embeddings = get_embeddings
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        self.embedding_api_token = embedding_api_token
        self.chunking_strategy = chunking_strategy
        self.chunk_multipage = chunk_multipage
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store = store
        self.store_settings = store_settings
        self.store_name = store_name

    def get_data(self):
        if self.re_download:
            print('Downloading data from the specified notion pages and databases....')
            napr = NotionAPIPageReader(self.notion_api_key)
            if len(self.database_ids) == 1:
                documents = napr.load_data(page_ids=self.page_ids, database_id=self.database_ids[0])

            else:
                documents = []
                documents.extend(napr.load_data(page_ids=self.page_ids))
                for database_id in self.database_ids:
                    documents.extend(napr.load_data(database_id=database_id))

            save_file_name = os.path.join(os.path.abspath(self.download_dir),
                                          'notion_data' + str(time.strftime("%Y%m%d-%H%M%S") + '.json'))
            if not os.path.exists(os.path.dirname(save_file_name)):
                os.makedirs(os.path.dirname(save_file_name))
            with open(save_file_name, 'w') as fp:
                json.dump(napr.json_data, fp)

            if self.store == 'simple':
                docstore = SimpleDocumentStore()
                docstore.add_documents(documents)
                docstore.persist(self.store_settings['persistent_path']+'/docstore.json')
            return documents
        else:
            docstore = SimpleDocumentStore().from_persist_dir(self.store_settings['docstore_path'])
            return docstore

    def de_dupe(self, documents: list = []):
        #ToDo: implement this function if the data needs to be de duped.
        if len(documents) == 0:
            raise ValueError('No documents are received from the NotionPageReader')


if __name__ == '__main__':
    gdfnli = GetDataFromNotionAPI(notion_api_key=os.environ['notion_api_key'],
                                  database_ids=['ce7b2781bb6e41b2864f0a59229ac781'])
    # page_ids=['0ce1e59f928d4109b690ba8f8a05519d'])
    # database_ids=['b9dacdd497314ebfae96b93b4e9332ac'])

    gdfnli.get_data()
