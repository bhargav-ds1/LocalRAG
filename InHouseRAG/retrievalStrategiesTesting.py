from DataHouse.Readers import GetDataFromNotion
from DataHouse.Cleaners import NotionDataCleaner
from EmbeddingStore import GetAndStoreEmbeddings
from QueryEngines import GetQueryEngine
from Retrievers import GetRetriever
from Configs import RetrievalQueryEngineNotionReciprocalRerankFusionRetriever
from dotenv import load_dotenv
import os
from Observability import InitializeObservability
from Evaluation import MakeRAGEvaluationDataset, Evaluation


class RAGBasedOnPaulGrahamData:
    def __init__(self):
        self.root_dir = os.path.dirname(__file__)
        InitializeObservability()
        load_dotenv(self.root_dir + '/.env-notion')

    def getData(self, data_args: dict = None) -> None:
        # Import notion data from notion using unstructured-io

        if 'notion_api_key' in os.environ.keys():
            gdn = GetDataFromNotion(notion_api_key=os.environ['notion_api_key'], **data_args)
            gdn.getData()
        else:
            raise ValueError(
                'Notion API key is required to get the data. Provide it by setting \"notion_api_value\" value in '
                'the environment variables or from a valid env file.')

    def cleanData(self, output_dir: str, items_to_remove: list):
        cdn = NotionDataCleaner(output_dir=output_dir, items_to_remove=items_to_remove)
        cdn.remove_element_from_json()

    def storeEmbeddings(self, input_dir: str, store: str):
        gse = GetAndStoreEmbeddings(dir_path=input_dir, store=store)
        gse.getEmbeddings()

    def make_query_engine(self, retriever_args=None, llm_args=None,
                          query_engine_args=None,
                          ):
        gr = GetRetriever(**retriever_args,**llm_args)
        retriever = gr.getRetriever()
        gqe = GetQueryEngine(retriever=retriever, **query_engine_args)
        query_engine = gqe.getQueryEngine()
        return query_engine

    def makeEvaluationDataset(self, query_engine, rag_dataset_generation_args):
        ged = MakeRAGEvaluationDataset(query_engine=query_engine, **rag_dataset_generation_args)
        ged.getEvaluationDataset()
        ged.saveDataset()

    def evaluate(self,query_engine, evaluation_args):
        evaluation = Evaluation(query_engine=query_engine,**evaluation_args)
        evaluation.evaluate()



def get_query_engine(arguments_config:dict=None):
    notion_model = RAGBasedOnPaulGrahamData()
    #notion_model.getData(data_args=arguments_config['data_args'])

    query_engine = notion_model.make_query_engine(
        retriever_args=arguments_config['retriever_args'],
        llm_args=arguments_config['llm_args'],
        query_engine_args=arguments_config['query_engine_args']
    )
    #notion_model.makeEvaluationDataset(query_engine=query_engine,
    #                                   rag_dataset_generation_args=arguments_config['rag_dataset_generation_args'])
    notion_model.evaluate(query_engine=query_engine,evaluation_args=arguments_config['evaluation_args'])

    return query_engine


if __name__ == '__main__':
    get_query_engine(arguments_config=RetrievalQueryEngineNotionReciprocalRerankFusionRetriever)
    # notion_model.make_streamlit_app(query_engine=query_engine)

    # notion_model.cleanData(output_dir='./DataHouse/Data/notion-ingest-output',items_to_remove=["UncategorizedText"])
    # notion_model.storeEmbeddings(input_dir='./DataHouse/Data/notion-ingest-output',
    #                             store='chroma')
