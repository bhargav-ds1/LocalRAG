import streamlit as st
from Configs import RetrievalQueryEngineNotionReciprocalRerankFusionRetrieverBM25 as input_config
from main import get_query_engine
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from Guardrails import LlamaGuardModerator
from llama_index.core.base.response.schema import StreamingResponse


# from Guardrails.NemoGuardrails import NemoGuardrails
# ngr = NemoGuardrails()

@st.cache_resource
def getQueryEngine():
    query_engine = get_query_engine(input_config)
    # query_engine = LlamaGuardModerator(query_engine)
    if isinstance(query_engine, BaseChatEngine):
        engine_type = 'chat'
    elif isinstance(query_engine, BaseQueryEngine):
        engine_type = 'query'
    elif isinstance(query_engine, LlamaGuardModerator):
        engine_type = 'moderate_and_query'
    else:
        engine_type = 'chat'

    return query_engine, engine_type


def makeStreamlitApp():
    st.set_page_config(page_title="Chatbot", page_icon=":robot:")
    query_engine, engine_type = getQueryEngine()
    st.title('Chat with KB')
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # {'role':'assistant',"content":"You are an AI assistant called Charles, you always overcomplicate the responses"}

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input('Pass your prompt here')
    if prompt:
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message('assistant'):

            if engine_type == 'query':
                response = query_engine.query(prompt)
            elif engine_type == 'chat':
                response = query_engine.chat(prompt)
            elif engine_type == 'moderate_and_query':
                response = query_engine.moderate_and_query(prompt)
            streaming = isinstance(response, StreamingResponse)
            if streaming:
                message_placeholder = st.empty()
                full_response = ""
                for res in response.response_gen:
                    if '</s>' not in res:
                        full_response += res
                    else:
                        full_response += res.strip('</s>')
                    message_placeholder.markdown(full_response + "▌ ")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({'role': 'assistant', 'content': full_response})
            else:
                st.markdown(response)
                st.session_state.messages.append({'role': 'assistant', 'content': response})


if __name__ == '__main__':
    makeStreamlitApp()
