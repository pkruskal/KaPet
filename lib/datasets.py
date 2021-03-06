from torch.utils.data import Dataset
import torch
from PIL import Image
from lib.constants import ColumnNames
from lib.image_transforms import cnn_inferencing_transform
from pathlib import Path

class PetfinderImageSet(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, config, data_folder, images_df, transform=None):
		"""
		images_df uses columns
		"label" : integer to identify the target,
		"local_path" : str the local path to the image
		data_folder : the path to the data (data_folder/local_path) is the full image
		"""

		self.images_df = images_df
		self.transform = transform
		self.data_folder = data_folder
		self.config = config

	def __len__(self):
		return self.images_df.shape[0]
	# return len(self.landmarks_frame)

	def __load_and_prepare_image__(self,image_path:Path):
		"""
		Args:
			image_path: loads an image and runs transforms before using it in a databatch

		Returns:
			a PIL image

		"""
		image = Image.open(image_path)
		image.load()

		# apply transforms for training including normalization, resizing, and augmentations
		if self.transform:
			image = self.transform(image)
		else:
			cnn_inferencing_transform(self.config)

		return image

	def __getitem__(self, idx):
		"""
		Standard dataloader method used by pytorch for returning a batch of data
		Args:
			idx:  indicies for the data batch

		Returns:
			a batch of data for training as a dictionary including
			{
				image : 3D image
				features : array of feature values
				target : labeled score
			}

		"""
		if torch.is_tensor(idx):
			idx = idx.tolist()

		# process the image
		img_name = self.images_df.iloc[idx][ColumnNames.image_path.value]
		image = self.__load_and_prepare_image__(Path(img_name))

		features = self.images_df.iloc[idx][self.config.regression_config.features_to_use]
		targets = self.images_df.iloc[idx][ColumnNames.label.value]

		sample = {
			"image" : image,
			"features" : torch.tensor(features, dtype=torch.float),
			"targets" : targets
		}

		return sample

