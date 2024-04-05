class DefaultIngest:
    embedding_providers = ['langchain-openai','langchain-huggingface','llama-index-huggingface','langchain-aws-bedrock']
    chunking_strategy = ['basic','by_title']
    storage_types = ['nebula','simple','azure','azure-cognitive-search','box','chroma','databricks-volumes', 'delta-table',
                              'dropbox', 'elasticsearch', 'fsspec', 'gcs', 'mongodb', 'opensearch', 'pinecone',
                              'qdrant', 's3', 'sql', 'vectara', 'weaviate']
