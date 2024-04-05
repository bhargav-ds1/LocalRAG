from llama_index.core.evaluation import RelevancyEvaluator, ContextRelevancyEvaluator, AnswerRelevancyEvaluator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core import SimpleDirectoryReader
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import ServiceContext
import time
import os

from llama_index.core.storage.docstore import SimpleDocumentStore


class MakeRAGEvaluationDataset:
    # TODO: make this class like from_documents like eg. ServiceContext.from_defaults
    def __init__(self, query_engine: BaseQueryEngine, raw_data_dir: str = None, doc_store_path: str = None,
                 required_keywords: list = None,
                 dataset_name: str = None,
                 output_dir: str = 'Evaluation/GeneratedRAGDataset', llm=None,
                 questions_per_chunk: int = 3):
        self.query_engine = query_engine
        self.service_context = query_engine._retriever.get_service_context()
        self.raw_data_dir = raw_data_dir
        self.doc_store_path = doc_store_path
        if not dataset_name.endswith('.json'):
            self.dataset_name = dataset_name + '.json'
        else:
            self.dataset_name = dataset_name
        self.llm = llm
        if self.llm is not None:
            self.service_context = ServiceContext.from_service_context(service_context=self.service_context, llm=llm)
        self.output_dir = output_dir
        self.questions_per_chunk = questions_per_chunk
        self.eval_questions = None
        self.root_dir = os.getcwd()
        self.output_dir = os.path.join(self.root_dir, self.output_dir)
        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)

    def getEvaluationDataset(self):
        if self.raw_data_dir is not None:
            documents = SimpleDirectoryReader(self.raw_data_dir).load_data()
        elif self.doc_store_path:
            docstore = SimpleDocumentStore().from_persist_dir(self.doc_store_path)
            documents = list(docstore.docs.values())
        else:
            raise ValueError('Provide raw_data_dir or doc_store_path from which to get the documents')
        data_generator = RagDatasetGenerator.from_documents(documents, service_context=self.service_context,
                                                            show_progress=True)
        self.eval_questions = data_generator.generate_questions_from_nodes()

    def saveDataset(self):
        if self.dataset_name is not None:
            self.eval_questions.save_json(self.output_dir + '/' + self.dataset_name)
        else:
            self.dataset_name = 'GeneratedDataset-' + str(time.strftime("%Y%m%d-%H%M%S"))
            self.eval_questions.save_json(self.dataset_name)

    def loadDataset(self):
        if self.output_dir:
            self.eval_questions = LabelledRagDataset.from_json(self.output_dir + '/' + self.dataset_name)
        else:
            raise ValueError(
                'output_dir argument is invalid, could not find the dataset specified by dataset_name in the output_dir')


'''
    def makeQueries(self):
        responses = []
        eval_questions = self.getEvaluationDataset()
        for question in eval_questions:
            full_response = ''
            response = self.query_engine.query(question)
            for res in response.response_gen:
                full_response += res
            responses.append(full_response)
'''
