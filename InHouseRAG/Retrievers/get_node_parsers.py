import os
from typing import Optional

import llama_index.core.node_parser as npar
from llama_index.core.callbacks import CallbackManager


class DefaultGetNodeParsers:
    available_node_parsers = ["TokenTextSplitter",
                              "SentenceSplitter",
                              "CodeSplitter",
                              "SimpleFileNodeParser",
                              "HTMLNodeParser",
                              "MarkdownNodeParser",
                              "JSONNodeParser",
                              "SentenceWindowNodeParser",
                              "SemanticSplitterNodeParser",
                              "NodeParser",
                              "HierarchicalNodeParser",
                              "TextSplitter",
                              "MarkdownElementNodeParser",
                              "MetadataAwareTextSplitter",
                              "LangchainNodeParser",
                              "UnstructuredElementNodeParser",
                              "get_leaf_nodes",
                              "get_root_nodes",
                              "get_child_nodes",
                              "get_deeper_nodes",
                              # deprecated, for backwards compatibility
                              "SimpleNodeParser", ]


class GetNodeParsers(DefaultGetNodeParsers):
    def __init__(self, node_parsers: dict = None, callback_manager: Optional[CallbackManager] = None):
        self.callback_manager = callback_manager
        self.node_parsers = node_parsers
        if self.node_parsers is not None:
            self.node_parsers_names = list(self.node_parsers.keys())

            if not all(x in self.available_node_parsers for x in self.node_parsers):
                raise ValueError('node_parsers should be one of :' + ','.join(self.available_node_parsers))

    def get_node_parsers(self):
        if self.node_parsers is None:
            return None
        node_pps = []
        for npp in self.node_parsers_names:
            if hasattr(npar, npp):
                if self.node_parsers[npp] is not None:
                    node_pps.append(getattr(npar, npp).from_defaults(**self.node_parsers[npp]))
                else:
                    node_pps.append(getattr(npar, npp).from_defaults())
        return node_pps
