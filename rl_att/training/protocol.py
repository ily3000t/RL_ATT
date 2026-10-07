"""Explicit production/engineering contracts for the PGD consistency baseline."""

from .state import require
from rl_att.defenses.pgd_consistency import validate_config as validate_defense


def validate_config(config):
    require(config["victim"] == "pgd_consistency" and config["protocol"] == "controlled" and
            config["gate_enabled"] is False and config["sumo_schedule"] == "derived" and config["cpu_threads"] == 1,
            "Only matched controlled CPU PGD consistency training without Gate")
    require(type(config["run_seed"]) is int and 0 <= config["run_seed"] < 5, "Use training seed 0 through 4")
    cli = config["training"]
    require(cli["seed"] == config["run_seed"] and (cli["state_dim"], cli["action_dim"], cli["action_numb"]) == (16, 1, 3) and
            cli["env"] == "highway-v0" and cli["mode"] == "train", "Preserve the SUMO observation/action training contract")
    require(type(cli["episodes"]) is int and cli["episodes"] >= 12 and type(cli["max_step"]) is int and cli["max_step"] > 0,
            "Training must exercise primary updates")
    require(config["training_stage"] in ("full", "engineering"), "Declare training stage")
    checkpoints = config["checkpoint_episodes"]
    require(isinstance(checkpoints, list) and checkpoints == sorted(set(checkpoints)) and
            all(type(value) is int and 0 < value <= cli["episodes"] for value in checkpoints) and
            checkpoints[-1] == cli["episodes"], "Declare ordered checkpoint episodes including the last episode")
    if config["training_stage"] == "full":
        require((cli["episodes"], cli["max_step"]) == (400, 200) and checkpoints == [100, 200, 300, 400],
                "Full baseline training is fixed at 400 x 200 with four checkpoints")
    else:
        require(cli["episodes"] <= 16 and cli["max_step"] <= 16, "Engineering training must stay short")
    require(config["update_recording"] == "all_gzip" and config["checkpoint_selection"] == "last_predeclared",
            "Record every update; never select a checkpoint by evaluation outcomes")
    defense = validate_defense(config["defense"])
    require(defense["coefficient"] > 0 and defense["epsilon"] > 0, "This training protocol exercises enabled PGD")
    return config
