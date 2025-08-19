# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Template version: v1.13.0
"""Unit tests for the `curobo_service` package version."""

# Standard Library
import pathlib

# NVIDIA
import nvidia.srl._curobo_service
import nvidia.srl.curobo_service


def test_srl_curobo_service_version() -> None:
    """Test that the `nvidia.srl.curobo_service` package version is set."""
    version = nvidia.srl._curobo_service._get_version()
    assert nvidia.srl.curobo_service.__version__ is not None
    assert nvidia.srl.curobo_service.__version__ == version


def test_srl_curobo_service_get_version_from_git_tag() -> None:
    """Test that the `_get_version_from_git_tag` function."""
    root = pathlib.Path(__file__).resolve().parent.parent
    version = nvidia.srl._curobo_service._get_version_from_git_tag(root.as_posix())
    # The version can be anything, just want to check that it returns something and doesn't error
    # out.
    assert version is not None


def test_srl_curobo_service_get_version_from_setuptools_scm_file() -> None:
    """Test that the `_get_version_from_setuptools_scm_file` function."""
    version = nvidia.srl._curobo_service._get_version_from_setuptools_scm_file()
    # The version can be anything, just want to check that it returns something and doesn't error
    # out.
    assert version is not None
