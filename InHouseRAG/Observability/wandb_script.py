from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.callbacks.wandb import WandbCallbackHandler
from llama_index.core import set_global_handler
from llama_index.core import Settings

set_global_handler("wandb", run_args={"project": "llamaindex"})
#wandb_callback = llama_index.core.global_handler

llama_debug = LlamaDebugHandler(print_trace_on_end=True)

# wandb.init args
run_args = dict(
    project="llamaindex",
)

wandb_callback = WandbCallbackHandler(run_args=run_args)

Settings.callback_manager = CallbackManager([llama_debug, wandb_callback])