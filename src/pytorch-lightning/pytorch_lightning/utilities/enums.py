# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Enumerated utilities."""
import os
from enum import Enum, EnumMeta
from typing import Any, List, Optional, Union

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.warnings import rank_zero_deprecation


class LightningEnum(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases."""

    @classmethod
    def from_str(cls, value: str) -> Optional["LightningEnum"]:
        statuses = [status for status in dir(cls) if not status.startswith("_")]
        for st in statuses:
            if st.lower() == value.lower():
                return getattr(cls, st)
        return None

    def __eq__(self, other: Union[str, Enum]) -> bool:
        other = other.value if isinstance(other, Enum) else str(other)
        return self.value.lower() == other.lower()

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        # example: set(LightningEnum)
        return hash(self.value.lower())


class _OnAccessEnumMeta(EnumMeta):
    """Enum with a hook to run a function whenever a member is accessed.

    Adapted from:
    https://www.buzzphp.com/posts/how-do-i-detect-and-invoke-a-function-when-a-python-enum-member-is-accessed
    """

    def __getattribute__(cls, name: str) -> Any:
        obj = super().__getattribute__(name)
        if isinstance(obj, Enum):
            obj.deprecate()
        return obj

    def __getitem__(cls, name: str) -> Any:
        member = super().__getitem__(name)
        member.deprecate()
        return member

    def __call__(cls, value: str, *args: Any, **kwargs: Any) -> Any:
        obj = super().__call__(value, *args, **kwargs)
        if isinstance(obj, Enum):
            obj.deprecate()
        return obj


class AMPType(LightningEnum):
    """Type of Automatic Mixed Precission used for training.

    >>> # you can match the type with string
    >>> AMPType.APEX == 'apex'
    True
    """

    APEX = "apex"
    NATIVE = "native"


class PrecisionType(LightningEnum):
    """Type of precision used.

    >>> PrecisionType.HALF == 16
    True
    >>> PrecisionType.HALF in (16, "16")
    True
    """

    HALF = "16"
    FLOAT = "32"
    FULL = "64"
    BFLOAT = "bf16"
    MIXED = "mixed"

    @staticmethod
    def supported_type(precision: Union[str, int]) -> bool:
        return any(x == precision for x in PrecisionType)

    @staticmethod
    def supported_types() -> List[str]:
        return [x.value for x in PrecisionType]


class DistributedType(LightningEnum, metaclass=_OnAccessEnumMeta):
    """Define type of training strategy.

    Deprecated since v1.6.0 and will be removed in v1.8.0.

    Use `_StrategyType` instead.
    """

    DP = "dp"
    DDP = "ddp"
    DDP2 = "ddp2"
    DDP_CPU = "ddp_cpu"
    DDP_SPAWN = "ddp_spawn"
    TPU_SPAWN = "tpu_spawn"
    DEEPSPEED = "deepspeed"
    HOROVOD = "horovod"
    DDP_SHARDED = "ddp_sharded"
    DDP_SHARDED_SPAWN = "ddp_sharded_spawn"
    DDP_FULLY_SHARDED = "ddp_fully_sharded"

    @staticmethod
    def interactive_compatible_types() -> List["DistributedType"]:
        """Returns a list containing interactive compatible DistributeTypes."""
        return [
            DistributedType.DP,
            DistributedType.DDP_SPAWN,
            DistributedType.DDP_SHARDED_SPAWN,
            DistributedType.TPU_SPAWN,
        ]

    def is_interactive_compatible(self) -> bool:
        """Returns whether self is interactive compatible."""
        return self in DistributedType.interactive_compatible_types()

    def deprecate(self) -> None:
        rank_zero_deprecation(
            "`DistributedType` Enum has been deprecated in v1.6 and will be removed in v1.8."
            " Use the string value `{self.value!r}` instead."
        )


class DeviceType(LightningEnum, metaclass=_OnAccessEnumMeta):
    """Define Device type by its nature - accelerators.

    Deprecated since v1.6.0 and will be removed in v1.8.0.

    Use `_AcceleratorType` instead.
    """

    CPU = "CPU"
    GPU = "GPU"
    IPU = "IPU"
    TPU = "TPU"

    def deprecate(self) -> None:
        rank_zero_deprecation(
            "`DeviceType` Enum has been deprecated in v1.6 and will be removed in v1.8."
            " Use the string value `{self.value!r}` instead."
        )


class GradClipAlgorithmType(LightningEnum):
    """Define gradient_clip_algorithm types - training-tricks.
    NORM type means "clipping gradients by norm". This computed over all model parameters together.
    VALUE type means "clipping gradients by value". This will clip the gradient value for each parameter.

    References:
        clip_by_norm: https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_
        clip_by_value: https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_value_
    """

    VALUE = "value"
    NORM = "norm"

    @staticmethod
    def supported_type(val: str) -> bool:
        return any(x.value == val for x in GradClipAlgorithmType)

    @staticmethod
    def supported_types() -> List[str]:
        return [x.value for x in GradClipAlgorithmType]


class AutoRestartBatchKeys(LightningEnum):
    """Defines special dictionary keys used to track captured dataset state with multiple workers."""

    PL_RESTART_META = "__pl_restart_meta"


class ModelSummaryMode(LightningEnum):
    # TODO: remove in v1.6 (as `mode` would be deprecated for `max_depth`)
    """Define the Model Summary mode to be used.

    Can be one of
        - `top`: only the top-level modules will be recorded (the children of the root module)
        - `full`: summarizes all layers and their submodules in the root module

    >>> # you can match the type with string
    >>> ModelSummaryMode.TOP == 'TOP'
    True
    >>> # which is case invariant
    >>> ModelSummaryMode.TOP in ('top', 'FULL')
    True
    """

    TOP = "top"
    FULL = "full"

    @staticmethod
    def get_max_depth(mode: str) -> int:
        if mode == ModelSummaryMode.TOP:
            return 1
        if mode == ModelSummaryMode.FULL:
            return -1
        raise ValueError(f"`mode` can be {', '.join(list(ModelSummaryMode))}, got {mode}.")

    @staticmethod
    def supported_types() -> List[str]:
        return [x.value for x in ModelSummaryMode]


class _StrategyType(LightningEnum):
    """Define type of training strategy.

    >>> # you can match the type with string
    >>> _StrategyType.DDP == 'ddp'
    True
    >>> # which is case invariant
    >>> _StrategyType.DDP2 in ('ddp2', )
    True
    """

    DP = "dp"
    DDP = "ddp"
    DDP2 = "ddp2"
    DDP_CPU = "ddp_cpu"
    DDP_SPAWN = "ddp_spawn"
    TPU_SPAWN = "tpu_spawn"
    DEEPSPEED = "deepspeed"
    HOROVOD = "horovod"
    DDP_SHARDED = "ddp_sharded"
    DDP_SHARDED_SPAWN = "ddp_sharded_spawn"
    DDP_FULLY_SHARDED = "ddp_fully_sharded"

    @staticmethod
    def interactive_compatible_types() -> List["_StrategyType"]:
        """Returns a list containing interactive compatible _StrategyTypes."""
        return [
            _StrategyType.DP,
            _StrategyType.DDP_SPAWN,
            _StrategyType.DDP_SHARDED_SPAWN,
            _StrategyType.TPU_SPAWN,
        ]

    def is_interactive_compatible(self) -> bool:
        """Returns whether self is interactive compatible."""
        return self in _StrategyType.interactive_compatible_types()


class _AcceleratorType(LightningEnum):
    """Define Accelerator type by its nature.

    >>> _AcceleratorType.CPU == _AcceleratorType.from_str('cpu')
    True
    >>> # you can match the type with string
    >>> _AcceleratorType.GPU == 'GPU'
    True
    >>> # which is case invariant
    >>> _AcceleratorType.TPU in ('tpu', 'CPU')
    True
    """

    CPU = "CPU"
    GPU = "GPU"
    IPU = "IPU"
    TPU = "TPU"


class _FaultTolerantMode(LightningEnum):

    DISABLED = "disabled"
    AUTOMATIC = "automatic"
    MANUAL = "manual"

    @property
    def is_enabled(self) -> bool:
        return self is not _FaultTolerantMode.DISABLED

    @property
    def is_automatic(self) -> bool:
        return self is _FaultTolerantMode.AUTOMATIC

    @property
    def is_manual(self) -> bool:
        return self is _FaultTolerantMode.MANUAL

    @classmethod
    def detect_current_mode(cls) -> "_FaultTolerantMode":
        """This classmethod detects if `Fault Tolerant` is activated and maps its value to `_FaultTolerantMode`."""
        env_value = os.getenv("PL_FAULT_TOLERANT_TRAINING", "0").lower()
        # the int values are kept for backwards compatibility, but long-term we want to keep only the strings
        if env_value in ("0", "disabled"):
            return _FaultTolerantMode.DISABLED
        elif env_value in ("1", "automatic"):
            return _FaultTolerantMode.AUTOMATIC
        elif env_value in ("2", "manual"):
            return _FaultTolerantMode.MANUAL
        raise MisconfigurationException(
            "The environment flag `PL_FAULT_TOLERANT_TRAINING` should be either 'disabled', 'automatic', or 'manual'."
        )
