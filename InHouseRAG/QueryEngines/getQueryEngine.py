from typing import Any, List, Optional, Sequence
from llama_index.core.settings import (Settings, callback_manager_from_settings_or_context)
from llama_index.core.response_synthesizers import BaseSynthesizer, ResponseMode, get_response_synthesizer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.callbacks.base import CallbackManager
from QueryEngines.get_prompt_templates import GetPromptTemplates
from QueryEngines.get_node_postprocessors import GetNodePostProcessors
import inspect
import sys
import logging


class DefaultGetQueryEngine:
    chat_engine_types = ["SimpleChatEngine",
                         "CondenseQuestionChatEngine",
                         "ContextChatEngine",
                         "CondensePlusContextChatEngine"]

    query_engine_types = ["CitationQueryEngine",
                          "CogniswitchQueryEngine",
                          "ComposableGraphQueryEngine",
                          "RetrieverQueryEngine",
                          "TransformQueryEngine",
                          "MultiStepQueryEngine",
                          "RouterQueryEngine",
                          "RetrieverRouterQueryEngine",
                          "ToolRetrieverRouterQueryEngine",
                          "SubQuestionQueryEngine",
                          "SQLJoinQueryEngine",
                          "SQLAutoVectorQueryEngine",
                          "RetryQueryEngine",
                          "RetrySourceQueryEngine",
                          "RetryGuidelineQueryEngine",
                          "FLAREInstructQueryEngine",
                          "PandasQueryEngine",
                          "JSONalyzeQueryEngine",
                          "JSONQueryEngine",
                          "KnowledgeGraphQueryEngine",
                          "BaseQueryEngine",
                          "CustomQueryEngine",
                          # multimodal
                          "SimpleMultiModalQueryEngine",
                          # SQL
                          "SQLTableRetrieverQueryEngine",
                          "NLSQLTableQueryEngine",
                          "PGVectorSQLQueryEngine",
                          ]
    response_modes = ['refine', 'compact', 'tree_summarize', 'simple_summarize', 'no_text', 'accumulate',
                      'compact_accumulate']
    prompt_template_providers = ['llama-index', 'langchain']


class GetQueryEngine(DefaultGetQueryEngine):
    def __init__(self, retriever,
                 query_engine_type: str = 'RetrieverQueryEngine',
                 query_engine_tools: dict = None,
                 query_engine_kwargs: dict = None,
                 response_mode: str = 'compact', streaming: bool = False,
                 prompt_template_provider: str = 'llama-index',
                 text_qa_template_str: str = None,
                 refine_template_str: str = None,
                 summary_template_str: str = None,
                 simple_template_str: str = None,
                 node_postprocessors: list[str] = None,
                 callback_manager: Optional[CallbackManager] = None, use_async: bool = False):

        super().__init__()
        self.query_engine_type = query_engine_type
        if self.query_engine_type not in self.query_engine_types and self.query_engine_type not in self.chat_engine_types:
            import QueryEngines as QE
            try:
                self.queryEngineClass = getattr(QE, self.query_engine_type)
            except Exception as e:
                raise ValueError('Query engine type should be one of :' + ','.join(
                    self.query_engine_types) + ' or should be a custom query engine implemented in the QueryEngines library')
        else:
            if self.query_engine_type in self.query_engine_types:
                import llama_index.core.query_engine as qen
                if hasattr(qen, self.query_engine_type):
                    self.queryEngineClass = getattr(qen, self.query_engine_type)
                elif self.query_engine_type == 'JSONQueryEngine':
                    import llama_index.core.indices.struct_store as ss
                    self.queryEngineClass = getattr(ss, self.query_engine_type)
            else:
                import llama_index.core.chat_engine as cen
                self.queryEngineClass = getattr(cen, self.query_engine_type)
        self.query_engine_tools = query_engine_tools
        self.query_engine_kwargs = query_engine_kwargs
        self.retriever = retriever
        self.response_mode = response_mode
        if self.response_mode not in self.response_modes:
            raise ValueError('Response mode should be one of :' + ','.join(self.response_modes))
        self.streaming = streaming
        if prompt_template_provider not in self.prompt_template_providers:
            raise ValueError('Prompt template provider should be one of :' + ','.join(self.prompt_template_providers))
        gpt = GetPromptTemplates(prompt_template_provider=prompt_template_provider,
                                 text_qa_template_str=text_qa_template_str, refine_template_str=refine_template_str,
                                 summary_template_str=summary_template_str, simple_template_str=simple_template_str)
        prompt_templates_dict = gpt.get_prompt_templates()
        self.text_qa_template = prompt_templates_dict['text_qa_template']
        self.refine_template = prompt_templates_dict['refine_template']
        self.summary_template = prompt_templates_dict['summary_template']
        self.simple_template = prompt_templates_dict['simple_template']

        self.callback_manager = callback_manager
        if self.callback_manager is None:
            self.callback_manager = callback_manager_from_settings_or_context(
                Settings, self.retriever.get_service_context())
        if node_postprocessors is not None:
            self.node_postprocessors = GetNodePostProcessors(node_postprocessors,
                                                             self.callback_manager).get_node_post_processors()
        else:
            self.node_postprocessors = node_postprocessors
        self.response_synthesizer = self.get_response_synthesizer()
        self.use_async = use_async

    def get_response_synthesizer(self) -> BaseSynthesizer:
        synthesizer = get_response_synthesizer(
            service_context=self.retriever.get_service_context(),
            text_qa_template=self.text_qa_template,
            refine_template=self.refine_template,
            summary_template=self.summary_template,
            simple_template=self.simple_template,
            response_mode=ResponseMode(self.response_mode),
            streaming=self.streaming,
            callback_manager=self.callback_manager
        )
        return synthesizer

    def getQueryEngine(self):
        query_engine_func = getattr(self, 'func_' + self.query_engine_type)
        queryEngine = query_engine_func()
        return queryEngine

    def func_FLAREInstructQueryEngine(self):
        from llama_index.core.query_engine import FLAREInstructQueryEngine
        #query_engine_func = getattr(self, 'func_' + self.query_engine_kwargs['query_engine_type'])
        #sub_query_engine = query_engine_func()
        sub_query_engine = self.retriever._index.as_query_engine()
        query_engine = FLAREInstructQueryEngine(query_engine=sub_query_engine,
                                                service_context=self.retriever.get_service_context(),
                                                max_iterations=self.query_engine_kwargs['max_iterations'],
                                                verbose=self.query_engine_kwargs['verbose'])
        return query_engine

    def func_JSONQueryEngine(self):
        from llama_index.core.indices.struct_store import JSONQueryEngine
        import json
        with open('./DataHouse/Data/Llama_index/rawData/notion_data20240302-200512.json') as fp:
            json_value = json.load(fp)
        with open('./DataHouse/Data/Llama_index/rawData/schema.json') as fp:
            json_schema = json.load(fp)
        return JSONQueryEngine(
            json_value=json_value,
            json_schema=json_schema,
            service_context=self.retriever.get_service_context()
        )

    def func_CondensePlusContextChatEngine(self):
        from llama_index.core.memory import ChatMemoryBuffer

        memory = ChatMemoryBuffer.from_defaults(token_limit=6400)
        return self.queryEngineClass.from_defaults(
            retriever=self.retriever,
            memory=memory,
            node_postprocessors=self.node_postprocessors,

        )

    def func_SubQuestionQueryEngine(self):
        # RetrieverQueryEngine based on VectorIndexRetriever is passed as a tool to the subquestion query engine
        query_engine_tools = [
            QueryEngineTool(
                # query_engine=self.func_RetrieverQueryEngine(args={'use_async':True}),
                query_engine=self.func_RetrieverQueryEngine(),
                metadata=ToolMetadata(
                    name=self.query_engine_tools['metadata']['name'],
                    description=self.query_engine_tools['metadata']['description'],
                ),
            ),
        ]
        return self.queryEngineClass.from_defaults(
            query_engine_tools=query_engine_tools,
            use_async=True,
            response_synthesizer=self.response_synthesizer,
            service_context=self.retriever.get_service_context()
        )

    def func_RetrieverQueryEngine(self):
        from llama_index.core.query_engine import RetrieverQueryEngine
        return RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            service_context=self.retriever.get_service_context(),
            response_synthesizer=self.response_synthesizer,
            node_postprocessors=self.node_postprocessors,
            callback_manager=self.callback_manager,
            use_async=self.use_async

        )

    def func_KnowledgeGraphQueryEngine(self):
        return self.retriever._index.as_query_engine()
        '''
        from llama_index.core.query_engine import KnowledgeGraphQueryEngine

        return KnowledgeGraphQueryEngine(
            storage_context=self.retriever._storage_context,
            service_context=self.retriever.get_service_context(),
            verbose=True,
            response_synthesizer=self.response_synthesizer
        )
        '''
