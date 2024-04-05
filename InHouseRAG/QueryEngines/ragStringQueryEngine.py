from typing import Optional, List
from llama_index.core import PromptTemplate
from llama_index.core.query_engine.custom import CustomQueryEngine
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    get_response_synthesizer,
)
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)


class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = []
    callback_manager: Optional[CallbackManager] = CallbackManager([])

    def apply_node_postprocessors(
            self, nodes: List, query_str: str
    ):
        if self.node_postprocessors is not None:
            for node_postprocessor in self.node_postprocessors:
                node_postprocessor.callback_manager = self.callback_manager
            for node_postprocessor in self.node_postprocessors:
                nodes = node_postprocessor.postprocess_nodes(
                    nodes, query_bundle=query_str
                )
        return nodes

    def retrieve(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        return self.apply_node_postprocessors(nodes, query_str=query_str)

    def custom_query(self, query_str: str):
        llm = llm_from_settings_or_context(Settings, self.retriever.get_service_context())
        self.callback_manager = self.callback_manager or callback_manager_from_settings_or_context(
            Settings, self.retriever.get_service_context())

        self.response_synthesizer = self.response_synthesizer or get_response_synthesizer(
            llm=llm,
            callback_manager=self.callback_manager
        )

        with self.callback_manager.event(
            CBEventType.QUERY, payload = {EventPayload.QUERY_STR: query_str}
        ) as query_event:
            nodes = self.retrieve(query_str)
            context_str = "\n\n".join([n.node.get_content() for n in nodes])
            if self.response_synthesizer._streaming:
                response = llm.stream_complete(
                    self.response_synthesizer._text_qa_template.format(context_str=context_str, query_str=query_str)
                    )
            else:
                response = llm.complete(
                    self.response_synthesizer._text_qa_template.format(context_str=context_str, query_str=query_str)
                )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

            return response


@property
def retriever(self) -> BaseRetriever:
    """Get the retriever object."""
    return self._retriever
