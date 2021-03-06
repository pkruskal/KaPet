from torchvision import transforms
from PIL import Image
from lib.configuration import Config
import numpy as np
from typing import Union, List

normalization_transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

class PetfinderListingsTranform:
	"""
	This transform mimicks the transform that is performed by the petfinder website for images that are listed.
	Uses `transforms.functional.resized_crop`
	"""

	def __init__(self, config):
		self.config = config

	def __call__(self, image):
		"""
		Crops the image taking the largerst centermost square segment.
		Then resizes the image for the neural network input

		Args:
			image:

		Returns:
			transform

		"""
		[width,height] = transforms.functional._get_image_size(image)

		# Get the coordinates for creating an image crop along the largest dimension used by pytorch's crop function
		if width > height:
			# crop the width by setting the "left" coordinate so the crop is centered
			top = 0
			left = np.floor((width-height)/2)
			# final sizes will be based on the height
			crop_height =height
			crop_width=height
		else:
			# crop the height by setting the "top" coordinate so the crop is centered
			top = np.floor((height-width)/2)
			left = 0
			# final sizes will be based on the width
			crop_height =width
			crop_width=width


		return transforms.functional.resized_crop(
			image,
			top = top,
			left = left,
			height =crop_height,
			width = crop_width,
			size = (self.config.image_dimension, self.config.image_dimension),
			interpolation = Image.BICUBIC
		)

def image_shaping_transform(
		config : Config
) -> Union[PetfinderListingsTranform,transforms.Resize]:
	"""
	Args:
		config.cnn_config.image_dimension:
		config.cnn_config.shaping [resize,crop]:
			Resize warps the image to fit the nn size with all the information
			Crop cuts the image the same way petfinder does when it's displayed on the listing page

	Returns:
		a single transform either resizing or a custom croping based on petfinder

	"""
	if config.image_shaping.value == "resize":
		transform = transforms.Resize((config.image_dimension, config.image_dimension), Image.BICUBIC)
	elif config.image_shaping.value == "crop":
		transform = PetfinderListingsTranform(config)

	return transform

def transform_for_neural_network_formating() -> [transforms.ToTensor,transforms.Normalize]:
	"""
	Partial Transform to make sure the image is set up to be run by a NN
	Resizes the image by reshaping opposed to cropping or padding

	Useful for test data

	:param config:
	:return:
	List[
		transforms.ToTensor, make sure image is a tensor for processing with pyTorch
		transforms.Normalize normaliz the image for the NN
	]
	"""
	transform_list = [
		transforms.ToTensor(),
		normalization_transform
	]

	return transform_list

def augmentation_transforms(config : Config) -> [transforms.transforms.RandomTransforms, ...]:
	"""
	Tranforms the image for use by a CNN
	also adds image augmentations for rotations and flips
	resizes the image by reshaping opposed to cropping or padding

	:param config:
	:return:
	"""
	transform_list = [
		transforms.RandomAffine(
			degrees=config.augmentations.rotation_augmentations,
			translate=config.augmentations.translation_augmentations
		)
	]
	if config.augmentations.image_flips:
		transform_list.append(transforms.RandomHorizontalFlip())

	return transform_list

def cnn_training_transform(config : Config) -> transforms.Compose:
	"""
	Full set of transforms to prepair data for TRAINING a CNN from the configuration settings

	Args:
		config: configuration class

	Returns:
		a transform

	"""
	transforms_to_run = []
	transforms_to_run.extend(augmentation_transforms(config))
	transforms_to_run.append(image_shaping_transform(config))
	transforms_to_run.extend(transform_for_neural_network_formating())
	transform_compose = transforms.Compose(transforms_to_run)

	return transform_compose

def cnn_inferencing_transform(config : Config) -> transforms.Compose:
	"""
	Full set of transforms to prepare data for INFERENCING on a CNN from the configuration settings

	Args:
		config: configuration class

	Returns:
		a transform

	"""

	transforms_to_run = []
	transforms_to_run.append(image_shaping_transform(config))
	transforms_to_run.extend(transform_for_neural_network_formating())
	transform = transforms.Compose(transforms_to_run)

	return transform