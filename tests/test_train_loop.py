"""The train and evaluate scripts run end to end with the random agent."""

import json

import pytest

from gdrl import evaluate, train
from gdrl.config import apply_override, load_config


def test_config_overrides():
    cfg = load_config("configs/mock_random.yaml", ["train.total_steps=123", "env.level_ids=[7, 8]"])
    assert cfg["train"]["total_steps"] == 123
    assert cfg["env"]["level_ids"] == [7, 8]
    apply_override(cfg, "agent.rollout_steps=64")
    assert cfg["agent"]["rollout_steps"] == 64


def test_non_gd_env_drops_gd_defaults():
    cfg = load_config("configs/cartpole_random.yaml")
    assert cfg["env"] == {"id": "CartPole-v1"}


@pytest.mark.parametrize("config", ["configs/mock_random.yaml", "configs/cartpole_random.yaml"])
def test_train_and_evaluate_smoke(tmp_path, config):
    run_dir = train.main(["--config", config, "--set", "train.total_steps=1024",
                          "--set", f"train.runs_dir={tmp_path}", "--set", "train.tensorboard=false"])
    assert (run_dir / "config.yaml").exists()
    rows = (run_dir / "metrics.csv").read_text().splitlines()
    assert len(rows) > 1

    results = evaluate.main(["--run", str(run_dir), "--episodes", "2"])
    assert results["levels"]
    assert json.loads((run_dir / "eval.json").read_text()) == results
