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
"""nvidia-srl-curobo-service distribution version info."""

DISTRIBUTION_NAME = "nvidia_srl_curobo_service"

# If true, disables printing the warning message when the fallback version is used.
NO_WARN = True


# NOTE (roflaherty): This is inspired by how matplotlib generates its version value.
# https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/__init__.py#L161
def _get_version() -> str:
    """Return the version string used for __version__."""
    # Standard Library
    import pathlib

    # Get path to project root (i.e. the directory where the .git folder lives)
    root = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent
    try:
        if (root / ".git").exists() and not (root / ".git/shallow").exists():
            this_version = _get_version_from_git_tag(root.as_posix())
        else:  # Get the version from the _version.py setuptools_scm file.
            this_version = _get_version_from_setuptools_scm_file()
    except Exception as err:  # noqa: F841
        # NOTE (roflaherty): This try-except is used to catch any errors that are thrown while
        # trying to obtain the version value, display a user warning about the failure in trying to
        # obtain the version value, and set the version to an arbitrary value.
        this_version = _get_fallback_version(no_warn=NO_WARN)

    return this_version


def _get_version_from_git_tag(root: str) -> str:
    """Return the version string based on the git commit and the latest git tag."""
    # Third Party
    import setuptools_scm

    this_version: str
    # See the `setuptools_scm` documentation for the description of the schemes used below.
    # https://pypi.org/project/setuptools-scm/
    # NOTE: If these values are updated, they need to be also updated in `pyproject.toml`.
    this_version = setuptools_scm.get_version(
        root=root,
        version_scheme="no-guess-dev",
        local_scheme="dirty-tag",
    )
    return this_version


def _get_version_from_setuptools_scm_file() -> str:
    """Return the version string based on the latest installed package version."""
    try:
        # Standard Library
        from importlib.metadata import version
    except ModuleNotFoundError:
        # NOTE: `importlib.metadata` is provisional in Python 3.9 and standard in Python 3.10.
        # `importlib_metadata` is the back ported library for older versions of python.
        # Third Party
        from importlib_metadata import version

    this_version = version(DISTRIBUTION_NAME)
    return this_version


def _get_fallback_version(no_warn: bool = False) -> str:
    """Return an arbitrary version value and print a user warning.

    Args:
        no_warn: If true, the warning message is not printed.
    """
    # Standard Library
    import traceback
    import warnings
    from typing import Optional, Type, Union

    # Arbitrary version value
    this_version = "0.0.0.dev0"

    def warning_on_one_line(
        message: Union[Warning, str],
        category: Type[Warning],
        filename: str,
        lineno: int,
        line: Optional[str] = None,
    ) -> str:
        return "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)

    warnings.formatwarning = warning_on_one_line
    warn_msg = (
        "The following error was caught while trying to obtain the version value for the"
        f" '{DISTRIBUTION_NAME}' distribution. The version value will be set to the arbitrary"
        f" value of '{this_version}'"
    )

    if not no_warn:
        warnings.warn(warn_msg)
        traceback.print_exc()

    return this_version
