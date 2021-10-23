from enum import Enum
from typing import List
from pydantic import BaseModel
import yaml
from pathlib import Path

class LearningRateConfig(BaseModel):
	"""
	Args:
		epocs: (List[int])
		rates: (List[float])
	"""
	epocs: List[int]
	rates: List[float]


class CNNConfig(BaseModel):
	"""
	Args:
		model_configuraton: (str) model configuration
		batchsize: (int) batch size
		epocs: (int) epocs
		learning_rate: (LearningRateConfig)
	"""
	model_configuraton: str
	batch_size: int
	epocs: int
	learning_rate: LearningRateConfig


class DeviceChoice(Enum):
	"""Choices for device"""
	cpu = 'cpu'
	cuda = 'cuda'


class Config(BaseModel):
	"""
	Args:
		image_dimension: (int) Image size in pixels, e.g. 128
		device: (str) Device to use. Options: "cpu", "cuda"
		rotation_augmentations: (int) +/- in degrees
		translation_augmentations: (int) % translation of image before resizing
		cnn_config: (CNNConfig) configuration for CNN
	"""
	image_dimension: int
	device: DeviceChoice
	rotation_augmentations: int
	translation_augmentations: int
	cnn_config: CNNConfig


def load_configuration(configuration_path : Path) -> Config:
	"""

	Args:
		configuration_path: (Path) path to the desired yaml file

	Returns:

	"""
	config_dict = {}
	with configuration_path.open() as file:
		config_dict = yaml.safe_load(file)

	config = Config.parse_obj(config_dict)
	return config