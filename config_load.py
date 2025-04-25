import yaml

def if_None(value):
    if isinstance(value, str):
        return None if value == 'None' else value

def convert_config_values(configs, convert_map):
    for key, conversion_fn in convert_map.items():
        if key in configs:
            configs[key] = conversion_fn(configs[key])
    return configs

convert_map = {
    "Display": bool,
    "batch_size": int,
    "cv_folds": int,
    "drop_neg": float,
    "drop_rate": float,
    "heads": int,
    "hidden_channels": int,
    "lr": float,
    "load_data": bool,
    "num_epochs": int,
    "random_seed": int,
    "repeat": int,
    "sample_rate" : float,
}

def get():
    with open("config.yaml", 'r') as f:
        configs = yaml.safe_load(f)
        configs = convert_config_values(configs, convert_map)
    return configs

if __name__ == "__main__":
    print(get())