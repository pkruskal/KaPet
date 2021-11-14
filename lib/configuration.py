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

class AugmentationsConfig(BaseModel):
	"""
	Args:
		image_flips: (bool) should we allow the image to be flipped
		rotation_augmentations: (int) degrees that image could be rotated
		translation_augmentations: (List[float,float]) percent that image could be shifted in x and y
	"""
	image_flips : bool
	rotation_augmentations: int
	translation_augmentations: List

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


class RegressionConfig(BaseModel):
	"""
	For paramaters for performing feature regression
	Args:
		features_to_use: (list[str]) model configuration
		regression_mode: (str)
	"""
	features_to_use : List[str]
	regression_mode : str

class DeviceChoice(Enum):
	"""Choices for device"""
	cpu = 'cpu'
	cuda = 'cuda'

class ImageShapings(Enum):
	"""
	Choices for image reshaping to fit the image_dimention.
	"resize" : standard appraoch where image is simple squished into the right shape with bicubic approximations
	"crop" : specific for petfinder which crops the image to mimick petfinders website
	"""
	resize = "resize"
	crop = "crop"

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
	image_shaping : ImageShapings
	device: DeviceChoice
	augmentations : AugmentationsConfig
	cnn_config: CNNConfig
	regression_config: RegressionConfig


def load_configuration(configuration_path : Path):
	config_dict = {}
	with configuration_path.open() as file:
		config_dict = yaml.safe_load(file)

	config = Config.parse_obj(config_dict)
	return config