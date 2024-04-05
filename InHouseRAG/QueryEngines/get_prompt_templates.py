from llama_index.core.prompts import PromptTemplate


class GetPromptTemplates:
    def __init__(self, prompt_template_provider: str, text_qa_template_str: str = None,
                refine_template_str: str = None,
                summary_template_str: str = None,
                simple_template_str: str = None,
                ):
        self.prompt_template_provider = prompt_template_provider
        self.text_qa_template_str = text_qa_template_str
        self.refine_template_str = refine_template_str
        self.summary_template_str = summary_template_str
        self.simple_template_str = simple_template_str

    def get_prompt_templates(self):
        if self.prompt_template_provider == 'llama-index':
            # lazy import
            from llama_index.core.prompts import PromptTemplate
            # default as None because the response synthesizer replaces them with default prompts
            return {'text_qa_template': PromptTemplate(
                self.text_qa_template_str) if self.text_qa_template_str is not None else None,
                    'refine_template': PromptTemplate(
                        self.refine_template_str) if self.refine_template_str is not None else None,
                    'summary_template': PromptTemplate(
                        self.summary_template_str) if self.summary_template_str is not None else None,
                    'simple_template': PromptTemplate(
                        self.simple_template_str) if self.simple_template_str is not None else None
                    }
        elif self.prompt_template_provider == 'langchain':
            # TODO: Deal with prompt templates from langchain if needed
            from langchain import hub
            langchain_prompt = hub.pull("rlm/rag-prompt",api_url="https://api.hub.langchain.com")
            from llama_index.core.prompts import LangchainPromptTemplate

            lc_prompt_tmpl = LangchainPromptTemplate(
                template=langchain_prompt,
                template_var_mappings={"query_str": "question", "context_str": "context"},
            )
