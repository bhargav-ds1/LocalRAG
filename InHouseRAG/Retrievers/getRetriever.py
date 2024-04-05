from DataHouse.Readers import DefaultIngest
from Retrievers.getStore import GetStore
from llama_index.core import StorageContext, set_global_service_context
from llama_index.core import ServiceContext
import llama_index.core.indices as ind
import llama_index.core.retrievers as ret
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoModelForCausalLM
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.schema import Document, IndexNode
from Retrievers import GetNodeParsers
import inspect
import torch
import sys
import logging


class DefaultGetRetriever:
    def __init__(self):
        self.index_types = [
            "KeywordTableIndex",
            "SimpleKeywordTableIndex",
            "RAKEKeywordTableIndex",
            "SummaryIndex",
            "TreeIndex",
            "DocumentSummaryIndex",
            "KnowledgeGraphIndex",
            "PandasIndex",
            "VectorStoreIndex",
            "SQLStructStoreIndex",
            "MultiModalVectorStoreIndex",
            "EmptyIndex",
            "ComposableGraph",
            # legacy
            "GPTKnowledgeGraphIndex",
            "GPTKeywordTableIndex",
            "GPTSimpleKeywordTableIndex",
            "GPTRAKEKeywordTableIndex",
            "GPTDocumentSummaryIndex",
            "GPTListIndex",
            "GPTTreeIndex",
            "GPTPandasIndex",
            "ListIndex",
            "GPTVectorStoreIndex",
            "GPTSQLStructStoreIndex",
            "GPTEmptyIndex",

        ]
        self.retriever_types = ["VectorIndexRetriever",
                                "VectorIndexAutoRetriever",
                                "SummaryIndexRetriever",
                                "SummaryIndexEmbeddingRetriever",
                                "SummaryIndexLLMRetriever",
                                "KGTableRetriever",
                                "KnowledgeGraphRAGRetriever",
                                "EmptyIndexRetriever",
                                "TreeAllLeafRetriever",
                                "TreeSelectLeafEmbeddingRetriever",
                                "TreeSelectLeafRetriever",
                                "TreeRootRetriever",
                                "TransformRetriever",
                                "KeywordTableSimpleRetriever",
                                "BaseRetriever",
                                "RecursiveRetriever",
                                "AutoMergingRetriever",
                                "RouterRetriever",
                                "BM25Retriever",
                                "QueryFusionRetriever",
                                # SQL
                                "SQLRetriever",
                                "NLSQLRetriever",
                                "SQLParserMode",
                                # legacy
                                "ListIndexEmbeddingRetriever",
                                "ListIndexRetriever",
                                # image
                                "BaseImageRetriever",
                                # custom
                                "DenseXRetriever"]
        self.index_type = 'VectorStoreIndex'
        self.retriever_type = 'VectorIndexRetriever'
        self.stores = DefaultIngest.storage_types
        self.embedding_providers = DefaultIngest.embedding_providers
        self.llm_providers = ['langchain-openai', 'llama-index-huggingface', 'langchain-aws-bedrock']


class GetRetriever(DefaultGetRetriever):
    def __init__(self, llm_provider: str, llm_model_name: str, llm_model_path: str,
                 offload_dir: str = './offload_dir', cache_dir: str = None,
                 local_files_only: bool = False, context_window: int = 4096, max_new_tokens: int = 256,
                 generate_kwargs: dict = None, tokenizer_max_length: int = 4096,
                 stopping_ids: tuple[int] = (50278, 50279, 50277, 1, 0), index_type: str = 'VectorStoreIndex',
                 retriever_type: str = 'VectorIndexRetriever',
                 retriever_kwargs: dict = None,
                 retriever_tools: list = None,
                 node_parsers:dict = None,
                 create_embeddings: bool = False,
                 raw_data_dir: str = None, embedding_provider: str = 'langchain-huggingface',
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L12-v2',
                 embedding_api_token: str = None,
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 store: str = 'chroma', store_settings: dict = None):
        super().__init__()
        self.index_type = index_type
        if self.index_type not in self.index_types:
            raise ValueError('Index type should be one of :' + ','.join(self.index_types))
        self.retriever_type = retriever_type
        if self.retriever_type not in self.retriever_types:
            raise ValueError('Retriever type should be one of :' + ','.join(self.retriever_types))
        self.retriever_tools = retriever_tools
        self.retriever_kwargs = retriever_kwargs
        self.create_embeddings = create_embeddings
        self.store = store
        self.store_settings = store_settings
        self.getStore = GetStore(self.store, self.store_settings).getStore
        self.raw_data_dir = raw_data_dir
        if not self.create_embeddings:
            pass  # TODO: log a message that the raw_data_dir is being ignored as create_embeddings is False
        self.llm_provider = llm_provider
        if self.llm_provider not in self.llm_providers:
            raise ValueError('LLM provider should be one of :' + ','.join(self.llm_providers))
        self.llm_model_name = llm_model_name
        self.llm_model_path = llm_model_path
        self.offload_dir = offload_dir
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = generate_kwargs
        self.tokenizer_max_length = tokenizer_max_length
        self.stopping_ids = stopping_ids
        self.embedding_provider = embedding_provider
        if self.embedding_provider not in self.embedding_providers:
            raise ValueError('Embedding Provider should be one of :' + ','.join(self.embedding_providers))
        self.embedding_model_name = embedding_model_name
        self.embedding_model = self.getEmbeddingModel()
        self.embedding_api_token = embedding_api_token
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.node_parsers = GetNodeParsers(node_parsers=node_parsers).get_node_parsers()

    def getEmbeddingModel(self):
        if self.embedding_provider == 'langchain-openai':
            pass
        elif self.embedding_provider == 'langchain-huggingface':
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        elif self.embedding_provider == 'langchain-aws-bedrock':
            pass

    def get_llm_model(self):
        if self.llm_provider == 'langchain-openai':
            pass
        elif self.llm_provider == 'llama-index-huggingface':
            from llama_index.llms.huggingface import HuggingFaceLLM
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.llm_model_path,
                device_map="cpu",
                offload_folder=self.offload_dir,
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            llm = HuggingFaceLLM(
                context_window=self.context_window,
                max_new_tokens=self.max_new_tokens,
                generate_kwargs=self.generate_kwargs,
                # system_prompt=system_prompt,
                # query_wrapper_prompt=query_wrapper_prompt,
                tokenizer_outputs_to_remove=['</s>'],
                tokenizer_name=self.llm_model_name,
                model_name=self.llm_model_name,
                device_map="cpu",
                #stopping_ids=list(self.stopping_ids),
                tokenizer_kwargs={"max_length": self.tokenizer_max_length},
                model=model
                # uncomment this if using CUDA to reduce memory usage
                # model_kwargs={"torch_dtype": torch.float16}
            )
        elif self.llm_provider == 'langchain-aws-bedrock':
            pass

        return llm

    def getRetriever(self):
        Settings.llm = self.get_llm_model()
        Settings.embed_model = self.embedding_model
        retriever_func = getattr(self, 'func_' + self.retriever_type)
        retriever = retriever_func()
        return retriever

    def func_DenseXRetriever(self):
        pass
        '''
        from llama_index.core.retrievers import RecursiveRetriever
        nodes = .get_nodes_from_documents(documents)
        sub_nodes = self._gen_propositions(nodes)

        all_nodes = nodes + sub_nodes
        all_nodes_dict = {n.node_id: n for n in all_nodes}
        '''

    def func_AutoMergingRetriever(self):
        from llama_index.core.retrievers import AutoMergingRetriever
        storage_context, service_context, vectorStore = self.getContexts()
        vector_retriever = self.func_VectorIndexRetriever()
        retriever = AutoMergingRetriever(vector_retriever=vector_retriever, storage_context=storage_context,
                                         verbose=True)
        return retriever

    def func_RecursiveRetriever(self):
        storage_context, service_context, vectorStore = self.getContexts()
        node_parser = SentenceSplitter(chunk_size=self.chunk_size)
        docstore = SimpleDocumentStore().from_persist_dir(self.store_settings['docstore_path'])
        base_nodes = node_parser.get_nodes_from_documents(list(docstore.docs.values()))
        for idx, node in enumerate(base_nodes):
            node.id_ = f"node-{idx}"
        sub_chunk_sizes = [128, 256, 512]
        sub_node_parsers = [
            SentenceSplitter(chunk_size=c, chunk_overlap=0) for c in sub_chunk_sizes
        ]

        all_nodes = []
        for base_node in base_nodes:
            for n in sub_node_parsers:
                sub_nodes = n.get_nodes_from_documents([base_node])
                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                all_nodes.extend(sub_inodes)

            # also add original node to node
            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(original_node)
        all_nodes_dict = {n.node_id: n for n in all_nodes}
        from llama_index.core.indices import VectorStoreIndex
        # define recursive retriever
        vector_index_chunk = VectorStoreIndex(
            all_nodes, service_context=service_context
        )
        vector_retriever_chunk = vector_index_chunk.as_retriever(
            similarity_top_k=self.retriever_kwargs['similarity_top_k']
        )
        from llama_index.core.retrievers import RecursiveRetriever
        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever_chunk},
            node_dict=all_nodes_dict,
            verbose=True,
        )
        return recursive_retriever

    def func_VectorIndexRetriever(self):
        index = self.getIndex(index_type='VectorStoreIndex')
        return index.as_retriever(similarity_top_k=self.retriever_kwargs['similarity_top_k'])

    def func_BM25Retriever(self):
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.storage.docstore import SimpleDocumentStore
        if self.raw_data_dir is None and 'docstore_path' not in self.store_settings:
            raise ValueError(
                'raw_data_dir value is required. BM25Retriever needs docstore which is currently not accessible from '
                'an index. So access to the raw documents is needed to create a docstore.')
        if self.raw_data_dir:
            documents = SimpleDirectoryReader(self.raw_data_dir).load_data()
            docstore = SimpleDocumentStore()
            docstore.add_documents(documents)
        if 'docstore_path' in self.store_settings:
            docstore = SimpleDocumentStore().from_persist_dir(self.store_settings['docstore_path'])

        retriever = BM25Retriever.from_defaults(docstore=docstore, verbose=True)
        return retriever

    def func_DocumentSummaryIndexEmbeddingRetriever(self):
        from llama_index.core.indices.document_summary import (
            DocumentSummaryIndexEmbeddingRetriever,
        )
        index = self.getIndex('DocumentSummaryIndex')
        retriever = DocumentSummaryIndexEmbeddingRetriever(
            index=index,
            similarity_top_k=self.retriever_kwargs['similarity_top_k']
            # choice_select_prompt=None,
            # choice_batch_size=10,
            # choice_top_k=1,
            # format_node_batch_fn=None,
            # parse_choice_select_answer_fn=None,
        )
        return retriever

    def func_DocumentSummaryIndexLLMRetriever(self):
        from llama_index.core.indices.document_summary import (
            DocumentSummaryIndexLLMRetriever,
        )
        index = self.getIndex('DocumentSummaryIndex')
        retriever = DocumentSummaryIndexLLMRetriever(
            index=index
            # choice_select_prompt=None,
            # choice_batch_size=10,
            # choice_top_k=1,
            # format_node_batch_fn=None,
            # parse_choice_select_answer_fn=None,
        )
        return retriever

    def func_QueryFusionRetriever(self):
        from llama_index.core.retrievers import QueryFusionRetriever
        if self.retriever_tools is None:
            raise ValueError(
                'Retriever of type QueryFusionRetriever is chosen but no retrievers data is provided as retriever_tools')

            # lazy import
        retrievers = []

        for i in self.retriever_tools:
            retriever_func = getattr(self, 'func_' + i['retriever_type'])
            retriever_i = retriever_func()
            retrievers.append(retriever_i)

        retriever = QueryFusionRetriever(
            retrievers,
            similarity_top_k=self.retriever_kwargs['similarity_top_k'],
            num_queries=self.retriever_kwargs['num_queries'],
            mode=self.retriever_kwargs['mode'],
            use_async=self.retriever_kwargs['use_async'],
            verbose=self.retriever_kwargs['verbose'],
            query_gen_prompt=self.retriever_kwargs[
                'query_gen_prompt_str'] if 'query_gen_prompt_str' in self.retriever_kwargs.keys() else None
        )
        return retriever

    def func_RouterRetriever(self):
        storage_context, service_context, vectorStore = self.getContexts()
        if self.retriever_tools is None:
            raise ValueError(
                'Retriever of type RouterRetriever is chosen but no retrievers data is provided as retriever_tools')
            # lazy import
        from llama_index.core.tools import RetrieverTool
        retriever_tools = []

        for i in self.retriever_tools:
            retriever_func = getattr(self, 'func_' + i['retriever_type'])
            retriever = retriever_func()

            retriever_tool = RetrieverTool.from_defaults(retriever=retriever,
                                                         description=i['description']
                                                         )
            retriever_tools.append(retriever_tool)

        retriever = ret.RouterRetriever.from_defaults(
            retriever_tools=retriever_tools,
            service_context=service_context,
            select_multi=True,
        )
        return retriever

    def func_KnowledgeGraphRAGRetriever(self):
        from transformers import pipeline
        self.triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large',
                                          tokenizer='Babelscape/rebel-large', device='mps')

        index = self.getIndex(index_type='KnowledgeGraphIndex')
        # retriever_func = getattr(ret, self.retriever_type)
        # retriever = retriever_func(
        #    index=index,
        #    storage_context=self.storage_context,
        #    with_nl2graphquery=True,
        # )
        from llama_index.core.indices.knowledge_graph.retrievers import KGRetrieverMode
        return index.as_retriever(include_text=self.retriever_kwargs['include_text'],
                                  similarity_top_k=self.retriever_kwargs['similarity_top_k'],
                                  retriever_mode=KGRetrieverMode(self.retriever_kwargs['retriever_mode']))

    def func_KGTableRetriever(self):
        from transformers import pipeline
        self.triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large',
                                          tokenizer='Babelscape/rebel-large', device='mps')
        from llama_index.core.indices.knowledge_graph.retrievers import KGRetrieverMode
        index = self.getIndex(index_type='KnowledgeGraphIndex')
        retriever_func = getattr(ret, self.retriever_type)
        retriever = retriever_func(
            index=index,
            similarity_top_k=self.retriever_kwargs['similarity_top_k'],
            include_text=self.retriever_kwargs['include_text'],
            retriever_mode=KGRetrieverMode(self.retriever_kwargs['retriever_mode'])
        )
        return retriever

    def getContexts(self):
        if self.store == 'chroma':
            vectorStore = self.getStore()

        else:
            vectorStore = self.getStore()
        if isinstance(vectorStore, GraphStore):
            storage_context = StorageContext.from_defaults(graph_store=vectorStore, )
        else:
            storage_context = StorageContext.from_defaults(vector_store=vectorStore, )
        service_context = ServiceContext.from_defaults(llm=Settings.llm, embed_model=self.embedding_model,
                                                       transformations=self.node_parsers
                                                       )
        set_global_service_context(service_context)
        self.storage_context = storage_context
        return storage_context, service_context, vectorStore

    def getIndex(self, index_type=None):
        storage_context, service_context, vectorStore = self.getContexts()
        if self.store == 'chroma':
            count = vectorStore._collection.count()
        try:
            if index_type is None:
                indexClass = getattr(ind, str(
                    dict(inspect.signature(getattr(ret, self.retriever_type).__init__).parameters.items())['index'].
                    annotation)[:-2].split('.')[-1])
            else:
                indexClass = getattr(ind, index_type)
            Settings.embed_model = self.embedding_model
            if self.create_embeddings:
                if self.create_embeddings and self.raw_data_dir is not None:
                    documents = SimpleDirectoryReader(self.raw_data_dir).load_data()
                    index = indexClass.from_documents(documents=documents, storage_context=storage_context,
                                                      service_context=service_context)
                    return index
                elif self.create_embeddings and 'docstore_path' in self.store_settings:
                    docstore = SimpleDocumentStore().from_persist_dir(self.store_settings['docstore_path'])
                    index = indexClass.from_documents(documents=list(docstore.docs.values()),
                                                      storage_context=storage_context,
                                                      service_context=service_context,
                                                      include_embeddings=True,
                                                      kg_triplet_extract_fn=self.extract_triplets
                                                      )
                    index.storage_context.persist(self.store_settings['persistent_path'])
                    return index
            else:
                if not self.create_embeddings and 'persistent_path' in self.store_settings:
                    from llama_index.core import load_index_from_storage
                    # storage_context.persist(self.store_settings['persistent_path'])
                    index = load_index_from_storage(storage_context)
                    return index
                elif count != 0 and vectorStore.is_embedding_query:
                    index = indexClass.from_vector_store(vector_store=vectorStore)
                    return index
                else:
                    raise ValueError('create_embeddings is set to False but the vector store has no embeddings, '
                                     'set create_embeddings to True and provide the raw_data_dir along with '
                                     'the embedding_model_name and embedding_provider to create embeddings')

            # TODO: return initialised retriever from embeddings in store, check if store has embeddings

        except Exception as e:
            print(e)

    def extract_triplets(self, input_text):
        text = self.triplet_extractor.tokenizer.batch_decode(
            [self.triplet_extractor(input_text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])[0]
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and subject in input_text and relation != '' and relation in input_text and object_ != '' and object_ in input_text:
            triplets.append((subject.strip(), relation.strip(), object_.strip()))

        return triplets
