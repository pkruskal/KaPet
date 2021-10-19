from torch.utils.data import Dataset
import torch
from PIL import Image
from constants import ColumnNames

class Petfinder_dataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, data_folder, images_df, config, transform=None, ):
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

		# img_name = os.path.join(self.data_folder,self.images_df.iloc[idx]['local_path'])
		img_name = self.images_df.iloc[idx][ColumnNames.image_path.value]
		image = Image.open(img_name)

		image.load()
		# image = io.imread(img_name)

		if self.transform:
			image = self.transform(image)

		# ToDo add a check for the metadata columns being corect
		metadata =  self.images_df.iloc[idx][self.config["metadata_columns"]]

		labels = self.images_df.iloc[idx][ColumnNames.label.value]

		sample = {
			"image": torch.tensor(image, dtype=torch.float),
			"features": torch.tensor(metadata, dtype=torch.float),
			"labels": torch.tensor(labels, dtype=torch.float),
		}

		return sample

	def test_data(self):
		bad_data = []

		for idx in range(self.__len__()):
			if torch.is_tensor(idx):
				idx = idx.tolist()

			try:
				# img_name = os.path.join(self.data_folder,self.images_df.iloc[idx]['local_path'])
				img_name = self.images_df.iloc[idx][ColumnNames.image_path.value]
				image = Image.open(img_name)
				image.load()

				if self.transform:
					image = self.transform(image)

			except Exception as e:
				print(img_name, e)
				bad_data.append(img_name)

		if len(bad_data) == 0:
			print('all data passed')
		return bad_data


class PawpularDataset:
	def __init__(self, image_paths, dense_features, targets, augmentations):
		self.image_paths = image_paths
		self.dense_features = dense_features
		self.targets = targets
		self.augmentations = augmentations

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, item):
		image = cv2.imread(self.image_paths[item])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		if self.augmentations is not None:
			augmented = self.augmentations(image=image)
			image = augmented["image"]

		image = np.transpose(image, (2, 0, 1)).astype(np.float32)

		features = self.dense_features[item, :]
		targets = self.targets[item]

		return {
			"image": torch.tensor(image, dtype=torch.float),
			"features": torch.tensor(features, dtype=torch.float),
			"targets": torch.tensor(targets, dtype=torch.float),
		}