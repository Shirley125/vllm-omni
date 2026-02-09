import sys
from vllm.inputs.data import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest
from vllm.v1.request import RequestStatus

import vllm_omni.logger  # noqa: F401
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.rotary_embedding import OmniMRotaryEmbedding
from vllm_omni.request import OmniRequest

# Extend RequestStatus enum with omni-specific statuses
if not hasattr(RequestStatus, "WAITING_FOR_CHUNK"):
    member = RequestStatus._value2member_map_.get(-1)
    if member is None:
        member_type = getattr(RequestStatus, "_member_type_", object)
        if member_type is object:
            member = object.__new__(RequestStatus)
        else:
            member = member_type.__new__(RequestStatus, -1)
        member._name_ = "WAITING_FOR_CHUNK"
        member._value_ = -1
        RequestStatus._value2member_map_[-1] = member
        if hasattr(RequestStatus, "_member_names_"):
            RequestStatus._member_names_.append("WAITING_FOR_CHUNK")
    RequestStatus._member_map_["WAITING_FOR_CHUNK"] = member
    if "WAITING_FOR_CHUNK" not in RequestStatus.__dict__:
        type.__setattr__(RequestStatus, "WAITING_FOR_CHUNK", member)
for module_name, module in sys.modules.items():
    # only do patch on module of vllm, pass others
    if "vllm" not in module_name:
        continue
    if hasattr(module, "EngineCoreOutput") and module.EngineCoreOutput == _OriginalEngineCoreOutput:
        module.EngineCoreOutput = OmniEngineCoreOutput
    if hasattr(module, "EngineCoreOutputs") and module.EngineCoreOutputs == _OriginalEngineCoreOutputs:
        module.EngineCoreOutputs = OmniEngineCoreOutputs
    if hasattr(module, "TokensPrompt") and module.TokensPrompt == _OriginalTokensPrompt:
        module.TokensPrompt = OmniTokensPrompt
    if hasattr(module, "MRotaryEmbedding") and module.MRotaryEmbedding == _OriginalMRotaryEmbedding:
        module.MRotaryEmbedding = OmniMRotaryEmbedding
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest
