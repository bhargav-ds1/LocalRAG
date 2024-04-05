from typing import Optional
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import set_global_handler
from llama_index.core import Settings
import llama_index.core


class DefaultObservability:
    observ_provider = 'wandb'
    observ_providers = ['wandb', 'deepeval', 'simple','phoenix']


class InitializeObservability(DefaultObservability):
    def __init__(self, observ_provider: Optional[str] = 'wandb'):
        self.observ_provider = observ_provider
        if self.observ_provider not in self.observ_providers:
            raise ValueError('Observability provider should be one of ' + ','.join(self.observ_providers))
        if self.observ_provider == 'wandb':
            self.initializeWandb()
        if self.observ_provider == 'deepeval':
            self.initializeDeepEval()
        if self.observ_provider == 'simple':
            self.initializeSimple()
        if self.observ_provider == 'phoenix':
            self.initializePhoenix()

    def initializeWandb(self):
        set_global_handler("wandb", run_args={"project": "llamaindex"})
        #self.wandb_callback = llama_index.core.global_handler
        # Settings.callback_manager = CallbackManager([ wandb_callback])

        llama_debug = LlamaDebugHandler(print_trace_on_end=True)

        # wandb.init args
        run_args = dict(
            project="llamaindex",
        )

        #wandb_callback = WandbCallbackHandler(run_args=run_args)

        Settings.callback_manager = CallbackManager([llama_debug])

    def closeWandb(self):
        self.wandb_callback.finish()

    def initializeDeepEval(self):
        set_global_handler(eval_mode='deepeval')

    def initializeSimple(self):
        set_global_handler('simple')

    def initializePhoenix(self):
        import phoenix as px
        px.launch_app()
        llama_index.core.set_global_handler("arize_phoenix")
