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


class DefaultGetChatEngine:
    chat_engine_types = []
    response_modes = ['refine', 'compact', 'tree_summarize', 'simple_summarize', 'no_text', 'accumulate',
                      'compact_accumulate']
    prompt_template_providers = ['llama-index', 'langchain']


class GetQueryEngine(DefaultGetChatEngine):
    def __init__(self, retriever,
                 chat_engine_type: str = 'RetrieverQueryEngine',
                 chat_engine_tools: dict = None,
                 chat_engine_kwargs: dict = None,
                 response_mode: str = 'compact', streaming: bool = False,
                 prompt_template_provider: str = 'llama-index',
                 text_qa_template_str: str = None,
                 refine_template_str: str = None,
                 summary_template_str: str = None,
                 simple_template_str: str = None,
                 node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
                 callback_manager: Optional[CallbackManager] = None, use_async: bool = False):

        super().__init__()
        self.chat_engine_type = chat_engine_type
        if self.chat_engine_type not in self.chat_engine_types:
            import QueryEngines as QE
            try:
                self.queryEngineClass = getattr(QE, self.chat_engine_type)
            except Exception as e:
                raise ValueError('Query engine type should be one of :' + ','.join(
                    self.query_engine_types) + ' or should be a custom query engine implemented in the QueryEngines library')
        else:
            import llama_index.core.query_engine as qen
            self.queryEngineClass = getattr(qen, self.query_engine_type)
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
        self.node_postprocessors = GetNodePostProcessors(node_postprocessors,self.callback_manager).get_node_post_processors()
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

    def func_SubQuestionQueryEngine(self):
        query_engine_tools = [
            QueryEngineTool(
                # query_engine=self.func_RetrieverQueryEngine(args={'use_async':True}),
                query_engine=self.retriever._index.as_query_engine(),
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
        from llama_index.core.query_engine import KnowledgeGraphQueryEngine

        return KnowledgeGraphQueryEngine(
            storage_context=self.retriever._storage_context,
            service_context=self.retriever.get_service_context(),
            verbose=True,
            response_synthesizer=self.response_synthesizer
        )
