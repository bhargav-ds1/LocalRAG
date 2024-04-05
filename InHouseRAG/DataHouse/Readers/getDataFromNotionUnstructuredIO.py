from dotenv import load_dotenv
import os
import subprocess
import time
from DataHouse.Readers import DefaultIngest


class GetDataFromNotionUnstructuredIO(DefaultIngest):
    def __init__(self, notion_api_key: str, re_download: bool = False, recursive: bool = True, num_processes: int = 2,
                 verbose: bool = False,
                 output_dir: str = './DataHouse/Data/notion-ingest-output',
                 download_dir: str = './DataHouse/Data/Unstructured_IO/rawData',
                 work_dir: str = './DataHouse/Data/workDir',
                 page_ids: list = None, database_ids: list = None,
                 get_embeddings: bool = False,
                 embedding_provider: str = 'langchain-huggingface',
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L12-v2',
                 embedding_api_token: str = None,
                 chunking_strategy: str = 'basic', chunk_multipage: bool = True, chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 store: str = 'chroma', store_settings: dict = None, store_name: str = None
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
        self.page_ids = page_ids
        self.database_ids = database_ids
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

        if self.store in self.storage_types:
            self.store = store
        else:
            raise ValueError('Selected Storage type not available select from')

        if self.store_name is None: self.store_name = (str(self.store) + str(self.chunking_strategy) +
                                                       str(time.strftime("%Y%m%d-%H%M%S")))

    def get_data(self):
        if self.notion_api_key is not None:

            command = ('unstructured-ingest notion' + " --notion-api-key " + '\"' + self.notion_api_key + '\"' +
                       " --output-dir " + '\"' + str(self.output_dir) + '\"' +
                       " --num-processes " + str(self.num_processes)
                       )
            if self.page_ids is not None:
                ids = ''
                ids += ' --page-ids \"'
                ids += ', '.join([str(i) for i in self.page_ids])
                ids += '\" '
                command += ids
            if self.database_ids is not None:
                ids = ''
                ids += ' --database-ids \"'
                ids += ', '.join([str(i) for i in self.database_ids])
                ids += '\" '
                command += ids
            if self.re_download: command = command + ' --re-download'
            if self.download_dir: command += ' --download-dir ' + '\"' + self.download_dir + '\"'
            if self.work_dir: command += ' --work-dir ' + '\"' + self.work_dir + '\"'
            if self.recursive: command += ' --recursive'
            if self.verbose: command += ' --verbose '
            if self.get_embeddings:
                if self.embedding_provider: command += ' --embedding-provider ' + '\"' + self.embedding_provider + '\"'
                if self.embedding_model_name: command += ' --embedding-model-name ' + '\"' + self.embedding_model_name + '\"'
                # if self.embedding_api_token: command += ' --embedding-api-key '+'\"'+self.embedding_api_token+'\"'
                if self.chunking_strategy: command += ' --chunking-strategy ' + '\"' + self.chunking_strategy + '\"'
                if self.chunk_multipage: command += ' --chunk-multipage-sections'
                if self.chunk_size: command += ' --chunk-max-characters ' + str(self.chunk_size)
                if self.chunk_overlap: command += ' --chunk-overlap ' + str(self.chunk_overlap)
                if self.chunk_overlap: command += ' --chunk-overlap-all '
                if self.store: command += str(self.store)
                if self.store_settings: command += ' --settings ' + str(self.store_settings)
                if self.store_name: command += ' --collection-name ' + str(self.store_name)
                # TODO: connect to the chroma store using the settings or path (test)
                # command += ' --path \"./chroma_db\"'
                # command += ' --settings {\"persist_directory\":\"./chroma_db\"}'
                command += ' --host \"localhost\"'
                command += ' --port 8000'

            p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()
            print(out)

            # notionDataReader = NotionDataReader(notion_api_key=os.environ['notion_api_key'], re_download=True,
            #                                    page_ids=['af46eb50845443a993f9a427660243b4'])
            # notionDataReader.runner.run()
        else:
            raise ValueError('The API key for notion is missing. Set the "notion_api_key" value in the enviornment '
                             'variables or set in in the env file.')


'''
cmd = 'unstructured-ingest notion --output-dir notion-ingest-output --page-ids "af46eb50845443a993f9a427660243b4" --num-processes 2 --verbose --re-download --recursive --pdf-infer-table-structure'



p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
out, err = p.communicate()
result = out.split('')
for lin in result:
    if not lin.startswith('#'):
        print(lin)
'''
