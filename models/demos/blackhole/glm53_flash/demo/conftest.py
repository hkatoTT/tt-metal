# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The demo's command-line options, named as the other tt-metal demos name them (e.g. models/tt_transformers/demo).
Each overrides its environment variable (demo.py's docstring); an option another conftest of the same run registered
already is used as it is."""

OPTIONS = (
    ("--input_prompts", dict(action="store", help="JSON file of prompts: strings or {'prompt': ...} objects")),
    ("--max_generated_tokens", dict(action="store", type=int, help="new tokens a prompt at most")),
    ("--batch_size", dict(action="store", type=int, help="prompts to run: the file's repeated or cut to this many")),
    ("--max_seq_len", dict(action="store", type=int, help="the cache a user, in tokens")),
)


def pytest_addoption(parser):
    for name, kwargs in OPTIONS:
        try:
            parser.addoption(name, **kwargs)
        except ValueError:  # (already registered by another conftest in this run: the same meaning, reused)
            pass
