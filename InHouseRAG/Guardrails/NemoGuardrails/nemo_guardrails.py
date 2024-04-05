from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.providers import register_llm_provider

class NemoGuardrails:
    def __init__(self,config_path:str = 'Guardrails/NemoGuardrails/Config'):
        self.config_path = config_path
        self.rails_config = RailsConfig.from_path(self.config_path)
        self.rails = LLMRails(config=self.rails_config)

    def print_info(self):
        info = self.rails.explain()
        info.print_llm_calls_summary()
        print(info.colang_history)
