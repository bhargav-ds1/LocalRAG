# Configuration for the retrieval query engine for notion data

RecursiveRetrieverQE = {
    'data_args': {'output_dir': 'DataHouse/Data/Llama_index/notion-ingest-output',
                  'download_dir': 'DataHouse/Data/Llama_index/rawData/json_responses',
                  'work_dir': 'DataHouse/Data/Llama_index/workDir',
                  'page_ids': None,
                  'database_ids': ['ce7b2781bb6e41b2864f0a59229ac781'],
                  're_download': False, 'verbose': True, 'recursive': True, 'get_embeddings': False,
                  'embedding_provider': 'langchain-huggingface',
                  'embedding_model_name': 'sentence-transformers/all-MiniLM-L12-v2',
                  'chunking_strategy': 'basic', 'chunk_multipage': True, 'chunk_size': 1000, 'chunk_overlap': 200,
                  'store': 'simple', 'store_settings': {'persistent_path': 'DataHouse/Data/Llama_index/Storage',
                                                        'docstore_path': 'DataHouse/Data/Llama_index/rawData/documents'},
                  'store_name': 'docstore.json'
                  },
    'retriever_args': {'index_type': 'VectorStoreIndex', 'retriever_type': 'RecursiveRetriever',
                       'retriever_kwargs': {'similarity_top_k': 2, 'verbose': True},
                       'create_embeddings': True,
                       'raw_data_dir': None,
                       'embedding_provider': 'langchain-huggingface',
                       'embedding_model_name': 'sentence-transformers/all-MiniLM-L12-v2',
                       'chunk_size': 1024,
                       'chunk_overlap': 200, 'store': 'simple',
                       'store_settings': {'persistent_path': 'DataHouse/Data/Llama_index/Storage',
                                          'docstore_path': 'DataHouse/Data/Llama_index/rawData/documents'
                                          }},
    'llm_args': {'llm_provider': 'llama-index-huggingface',
                 'llm_model_name': 'meta-llama/Llama-2-7b-chat-hf',
                 'llm_model_path': '/Users/bhargavvankayalapati/Work/InHouseRAG/InHouseRAG/Models/meta-llama/Llama-2-7b-chat-hf',
                 'offload_dir': './offload_dir',
                 'cache_dir': '/Users/bhargavvankayalapati/.cache',
                 'local_files_only': True, 'context_window': 4096,
                 'max_new_tokens': 512,
                 'generate_kwargs': {"temperature": 0.7, "top_k": 50, "top_p": 0.95,
                                     'do_sample': False},
                 'tokenizer_max_length': 4096,
                 'stopping_ids': (50278, 50279, 50277, 1, 0),
                 },
    'query_engine_args': {'query_engine_type': 'RetrieverQueryEngine',
                          'query_engine_kwargs': None,
                          'response_mode': 'compact', 'streaming': True,
                          'text_qa_template_str': None,
                          'refine_template_str': None, 'summary_template_str': None,
                          'simple_template_str': None,
                          'node_postprocessors': {'SimilarityPostprocessor': {'similarity_cutoff': 0.5}},
                          'callback_manager': None,
                          'use_async': False},
    'rag_dataset_generation_args': {'raw_data_dir': 'DataHouse/Data/Unstructured_IO/rawData',
                                    'dataset_name': 'generated-RAG-dataset-notion-nodes',
                                    'questions_per_chunk': 10},
    'evaluation_args': {'project_id': '9500d44e-8059-46a5-9fe1-c4009715644b',
                        'evaluation_dataset': 'Evaluation/GeneratedRAGDataset/sample-generated-RAG-dataset-notion-nodes.json',
                        'eval_output_dir':'Evaluation/EvaluationResults/RecursiveRetriever'}
}
