import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import SimpleVectorStore
from DataHouse.Readers import DefaultIngest
import os

#TODO: When adding new stores create the function name as ' def (store_name)Store ' where 'store_name' is the
# name of the store from among the available stores.
class GetStore:
    def __init__(self,store,store_settings):
        self.store = store
        self.store_settings = store_settings
        self.stores = DefaultIngest.storage_types
        if self.store not in self.stores: raise ValueError('the provided storage type is not available pick from the '
                                                           'following '+','.join(self.stores))
    def getStore(self):
        #TODO: check if this method can be called without calling the exact name eg. __call__
        try:
            func = getattr(self, self.store + 'Store')
        except Exception as e:
            raise NotImplementedError('Store type not implemented')
        return func(self.store_settings)


    def chromaStore(self, storage_settings):
        db2 = chromadb.PersistentClient(path=storage_settings['persistent_path'])
        chroma_collection = db2.get_or_create_collection(storage_settings['collection_name'])
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store

    def simpleStore(self, storage_settings):
        vector_store = SimpleVectorStore()
        #vector_store.persist(persist_path=storage_settings['persistent_path']+'/docstore.json')
        return vector_store

    def nebulaStore(self,storage_settings):
        from llama_index.graph_stores.nebula import NebulaGraphStore
        graph_store = NebulaGraphStore(
            space_name=storage_settings['space_name'],
            edge_types=storage_settings['edge_types'],
            rel_prop_names=storage_settings['rel_prop_names'],
            tags=storage_settings['tags'],
        )
        return graph_store

    def pineconeStore(self,storage_settings):
        from pinecone import Pinecone
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        try:
            pc.create_index(
                "quickstart-index",
                dimension=1536,
                metric="cosine"
            )
            pinecone_index = pc.Index("quickstart-index")
            vector_store = PineconeVectorStore(
                pinecone_index=pinecone_index,
                namespace="test",
            )
            return vector_store
        except Exception as e:
            # Most likely index already exists
            print(e)
            pass

