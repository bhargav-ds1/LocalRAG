import os
from typing import Optional

import llama_index.core.postprocessor as pop
from llama_index.core.callbacks import CallbackManager


class DefaultGetNodePostProcessors:
    available_node_post_processors = ["SimilarityPostprocessor",
                                      "KeywordNodePostprocessor",
                                      "PrevNextNodePostprocessor",
                                      "AutoPrevNextNodePostprocessor",
                                      "FixedRecencyPostprocessor",
                                      "EmbeddingRecencyPostprocessor",
                                      "TimeWeightedPostprocessor",
                                      "PIINodePostprocessor",
                                      "NERPIINodePostprocessor",
                                      "LLMRerank",
                                      "SentenceEmbeddingOptimizer",
                                      "SentenceTransformerRerank",
                                      "MetadataReplacementPostProcessor",
                                      "LongContextReorder",
                                      "CohereRerank",
                                      "LongLLMLinguaPostprocessor"]


class GetNodePostProcessors(DefaultGetNodePostProcessors):
    def __init__(self, node_post_processors: dict = None, callback_manager: Optional[CallbackManager] = None):
        self.callback_manager = callback_manager
        self.node_post_processors = node_post_processors
        self.node_post_processor_names = list(self.node_post_processors.keys())

        if not all(x in self.available_node_post_processors for x in self.node_post_processors):
            raise ValueError('node_postprocessors should be one of :' + ','.join(self.available_node_post_processors))

    def get_node_post_processors(self):
        node_pps = []
        for npp in self.node_post_processor_names:
            if hasattr(pop, npp):
                node_pps.append(
                    getattr(pop, npp)(**self.node_post_processors[npp]))
            else:
                if npp == 'CohereRerank':
                    from llama_index.postprocessor.cohere_rerank import CohereRerank
                    node_pps.append(
                        CohereRerank(**self.node_post_processors[npp], api_key = os.environ['COHERE_API_KEY']))
                if npp == 'LongLLMLinguaPostprocessor':
                    from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor
                    node_pps.append(
                        LongLLMLinguaPostprocessor()
                    )
        return node_pps
