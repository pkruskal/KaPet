from torchvision import transforms
from PIL import Image
from configuration import Config
import numpy as np

normalization_transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

class PetfinderListingsTranform:
	"""
	This transform mimicks the transform that is performed by the petfinder website for images that are listed.
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
		[width,height] = transforms.functional.get_image_size(image)
		if width > height:
			top = height
			left = np.floor(width-height/2)
			crop_height =height
			crop_width=height
		else:
			top = np.floor(height-width/2)
			left = width
			crop_height =width
			crop_width=width


		return transforms.functional.resized_crop(
			image,
			top = top,
			left = left,
			height =crop_height,
			width = crop_width,
			size = (self.config.cnn_config.image_dimention, self.config.cnn_config.image_dimention),
			interpolation = Image.BICUBIC
		)

def image_shaping_transform(config : Config):
	"""

	Args:
		config.cnn_config.image_dimension:
		config.cnn_config.shaping [resize,crop]:
			Resize warps the image to fit the nn size with all the information
			Crop cuts the image the same way petfinder does when it's displayed on the listing page

	Returns:

	"""
	if config.cnn_config.shaping == "resize":
		transform = transforms.Resize((config.cnn_config.image_dimension, config.cnn_config.image_dimension), Image.BICUBIC),
	elif config.cnn_config.shaping == "crop":
		transform = PetfinderListingsTranform(config)

	return transform

def transform_for_neural_network_formating():
	"""
	Partial Transform to make sure the image is set up to be run by a NN
	Resizes the image by reshaping opposed to cropping or padding

	Useful for test data

	:param config:
	:return:
	"""
	transform = transforms.Compose([
		transforms.ToTensor(),
		normalization_transform
	])

	return transform

def augmentation_transform(config : Config):
	"""
	Tranforms the image for use by a CNN
	also adds image augmentations for rotations and flips
	resizes the image by reshaping opposed to cropping or padding

	:param config:
	:return:
	"""
	transform = transforms.Compose([
		transforms.RandomAffine(
			degrees=config["rotation_augmentations"],
			translate=(0,0)),
		transforms.RandomHorizontalFlip()
	])

	return transform

def cnn_training_transform(config : Config):
	"""
	Full set of transforms to prepair data for TRAINING a CNN from the configuration settings

	Args:
		config: configuration class

	Returns:
		a transform

	"""

	transform = transforms.Compose([
		augmentation_transform(config),
		image_shaping_transform(config),
		transform_for_neural_network_formating()
	])

def cnn_inferencing_transform(config : Config):
	"""
	Full set of transforms to prepair data for INFERENCING on a CNN from the configuration settings

	Args:
		config: configuration class

	Returns:
		a transform

	"""

	transform = transforms.Compose([
		image_shaping_transform(config),
		transform_for_neural_network_formating()
	])

	return transform