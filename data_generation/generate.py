import argparse
import copy
import json

import tqdm
from generator.generator import EnvironmentDataGenerator


def run_env(config, connector_config):
    connector_class_name = connector_config["classname"]
    del connector_config["classname"]
    generator_config = connector_config["generator_config"]
    del connector_config["generator_config"]

    generator = EnvironmentDataGenerator(
        connector_class_name,
        connector_config,
        generator_config,
        config,
    )
    print(f"Generating {generator.name}...")
    generator.generate()
    print(f"Done with {generator.name}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_gen_retro.json")
    parser.add_argument("--max_workers", type=int, default=24)
    args = parser.parse_args()
    config = json.load(open(args.config))

    connector_configs = []
    if config["env"] == "coinrun":
        connector_config = config["connector_" + config["env"]]

        from generator.connector_coinrun import CoinRunConnector

        connector_config["classname"] = CoinRunConnector

        connector_configs.append(connector_config)

    elif config["env"] == "retro_act":
        connector_config_retro_act = config["connector_" + config["env"]]

        from generator.connector_retro_act import (
            GameData,
            RetroActConnector,
        )

        connector_config_retro_act["classname"] = RetroActConnector
        game_data = GameData(
            annotation_fpath=connector_config_retro_act["annotation_behavior_fpath"],
            control_annotation_fpath=connector_config_retro_act[
                "annotation_control_fpath"
            ],
        )

        selected_games = connector_config_retro_act["game"]
        motion_filter = connector_config_retro_act["motion"]
        view_filter = connector_config_retro_act["view"]
        genre_filter = connector_config_retro_act["genre"]
        platform_filter = connector_config_retro_act["platform"]
        action_map = connector_config_retro_act["action_map"]

        if action_map == "default":
            action_map = {
                "ACTION_JUMP": "jump",
                "DOWN": "none|crouch|climb",
                "UP": "climb|none",
                "LEFT": "left",
                "RIGHT": "right",
            }
            
        game_data.filter(action_map=action_map)
        game_filter = None if selected_games == "all" else selected_games

        selected_games = game_data.query(
            genre=genre_filter,
            motion=motion_filter,
            view=view_filter,
            game=game_filter,
            platform=platform_filter,
        )

        if len(selected_games) == 0:
            raise ValueError(
                f"No games found for genre={genre_filter}, motion={motion_filter}, view={view_filter}, platform={platform_filter}",
            )

        print(
            f"Found {len(selected_games)} games for genre={genre_filter}, motion={motion_filter}, view={view_filter}",
        )

        for i, game in enumerate(selected_games):
            connector_config_retro_act_temp = copy.deepcopy(connector_config_retro_act)
            connector_config_retro_act_temp["game"] = game
            # if "PowerPiggs" in game:
            connector_configs.append(connector_config_retro_act_temp)
    else:
        raise ValueError(f"Unknown environment {config['env']}")

    for connector_config in tqdm.tqdm(connector_configs):
        run_env(config, connector_config)

    # pool = multiprocessing.Pool(processes=min(len(connector_configs),args.max_workers))
    # pool.starmap(run_env, [(config, connector_config) for connector_config in connector_configs])
    # pool.close()
    # pool.join()
    print("Done")
