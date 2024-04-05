from llama_index.core.llama_dataset import download_llama_dataset

# download and install dependencies
rag_dataset, documents = download_llama_dataset(
    "PaulGrahamEssayDataset", "./paul_graham"
)
for i in range(2):
    pass