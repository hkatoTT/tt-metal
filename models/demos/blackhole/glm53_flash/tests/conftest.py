# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
from pathlib import Path

import pytest
from loguru import logger


@pytest.fixture
def pcc_thresholds():
    """The PCC floors of the device tests against `tt/`, keyed by test function name (pcc_thresholds.json)."""
    return json.loads(Path(__file__).with_name("pcc_thresholds.json").read_text())


@pytest.fixture
def expect_error():
    """The tree's `expect_error` (root conftest.py), for the tests run with --confcutdir here (no device): use it in
    place of pytest's `raises`; the log marks the error as expected."""

    @contextlib.contextmanager
    def expect_error_(error, message):
        names = ", ".join(e.__name__ for e in (error if isinstance(error, tuple) else (error,)))
        logger.info(f'[EXPECTED_ERROR BEGIN] {names} message="{message}"')
        try:
            with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture itself
                yield exc_info
        finally:
            logger.info(f'[EXPECTED_ERROR END] {names} message="{message}"')

    return expect_error_
