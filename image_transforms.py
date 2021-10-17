from torchvision import transforms
from PIL import Image


normalization_transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def augmentation_transform_pytorch(config : dict):
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
		transforms.RandomHorizontalFlip(),
		transforms.Resize((config["image_dimention"], config["image_dimention"]), Image.BICUBIC),
		#transforms.RandomCrop(res, padding=4),
		#transforms.CenterCrop(res),
		transforms.ToTensor(),
		normalization_transform
	])

	return transform

def pure_transform_pytorch(config : dict):
	"""
	Transforms the data for use in a neural network with no augmentations
	Resizes the image by reshaping opposed to cropping or padding

	Useful for test data

	:param config:
	:return:
	"""
	transform = transforms.Compose([
		transforms.Resize((config["image_dimention"], config["image_dimention"]), Image.BICUBIC),
		transforms.ToTensor(),
		normalization_transform
	])

	return transform
