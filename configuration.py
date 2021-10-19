from pathlib import Path
import yaml
import torch

def load_configuration(configure_path : Path) -> dict:
	with open(configure_path / "configuration.yaml","r") as f:
		config = yaml.load(f.read(),yaml.Loader)

	print(f'working on device {config["device"]}')

	config["device"] = torch.device(config["device"])

	return config

