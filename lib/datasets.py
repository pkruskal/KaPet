from torch.utils.data import Dataset
import torch
from PIL import Image
from constants import ColumnNames
from image_transforms import cnn_inferencing_transform

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

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		# process the image
		img_name = self.images_df.iloc[idx][ColumnNames.image_path.value]
		image = Image.open(img_name)
		image.load()

		if self.transform:
			image = self.transform(image)
		else:
			cnn_inferencing_transform(self.config)

		features = self.images_df.iloc[idx][self.config.regression_config.features_to_use]
		targets = self.images_df.iloc[idx][ColumnNames.label.value]


		sample = {
			"image" : image,
			"features" : torch.tensor(features, dtype=torch.float),
			"targets" : targets
		}

		return sample

