from __future__ import annotations

from pathlib import Path

import pytest

from .harness import create_spec_kitty_wrapper, install_fake_agents, make_base_env, seed_minimal_spec_kitty_project


@pytest.fixture()
def fake_agent_project(tmp_path: Path):
    bin_dir, wrapper_env = create_spec_kitty_wrapper(tmp_path)
    install_fake_agents(bin_dir)
    env = make_base_env(bin_dir, wrapper_env)
    return seed_minimal_spec_kitty_project(tmp_path, bin_dir=bin_dir, env=env)


@pytest.fixture()
def real_agent_project(tmp_path: Path):
    bin_dir, wrapper_env = create_spec_kitty_wrapper(tmp_path)
    env = make_base_env(bin_dir, wrapper_env)
    return seed_minimal_spec_kitty_project(tmp_path, bin_dir=bin_dir, env=env)
