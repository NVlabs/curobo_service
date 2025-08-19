# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Template version: v1.11.0
"""NVIDIA SRL Curobo Service package."""


# NVIDIA
from nvidia.srl._curobo_service import _get_version

# Set `__version__` attribute
__version__ = _get_version()

# Remove `_get_version` so it is not added as an attribute
del _get_version
