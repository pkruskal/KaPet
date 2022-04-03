import timm
from torch import nn
from .base_image_models import BaseRegressionModel
from .base_image_models import BaseClassificationModel

class PawpularSwinModel(BaseRegressionModel):
	"""
	This Swin model makes use of the torch image models module (timm)

	For more info on the Swin image recognition framework
	https://arxiv.org/pdf/2103.14030.pdf

	To see valid model_names try
	[m for m in timm.list_models() if "swin" in m]
	e.x.
		"swin_base_patch4_window7_224"
		"swin_base_patch4_window12_384"
		"swin_large_patch4_window7_224"


	For more information on these configurations see
	https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
	https://github.com/microsoft/Swin-Transformer/tree/main/configs

	See BaseRegressionModel for documentation on this class

	"""

	def __init__(
			self,
			model_name: str,
			learning_rate: float,
			number_of_latent_image_features: int = 128,
			number_of_additional_features: int = 12,
			number_of_intermediate_regression_variables: int = 64,
			regression_dropout: float = 0.1,
			regression_activation_function: nn.Module = nn.Identity(),
			image_size: int = 384,
			model_drop_rate: float = 0
	):
		"""

		:param image_size: (int) for initializeing the swin model we need to state the image size explicitly here.
			Note different swin models expect different image sizes using timm
		"""
		super().__init__(
			model_name=model_name,
			learning_rate=learning_rate,
			number_of_latent_image_features=number_of_latent_image_features,
			number_of_additional_features=number_of_additional_features,
			number_of_intermediate_regression_variables=number_of_intermediate_regression_variables,
			regression_dropout=regression_dropout,
			regression_activation_function=regression_activation_function
		)

		self.model = timm.create_model(
			model_name,
			image_size = image_size,
			pretrained=False,
			in_chans=3,
			num_classes = number_of_latent_image_features,
			model_drop_rate=model_drop_rate
		)

		self.learning_rate = learning_rate



class PawpularSwinClassificationModel(BaseClassificationModel):
	"""
	This Swin model makes use of the torch image models module (timm)

	For more info on the Swin image recognition framework
	https://arxiv.org/pdf/2103.14030.pdf

	To see valid model_names try
	[m for m in timm.list_models() if "swin" in m]
	e.x.
		"swin_base_patch4_window7_224"
		"swin_base_patch4_window12_384"
		"swin_large_patch4_window7_224"


	For more information on these configurations see
	https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
	https://github.com/microsoft/Swin-Transformer/tree/main/configs

	See BaseRegressionModel for documentation on this class

	"""

	def __init__(
			self,
			model_name: str,
			learning_rate: float,
			bins,
			number_of_classes: int = 128,
			image_size: int = 384,
			model_drop_rate: float = 0,
			loss_function: nn.Module = nn.BCEWithLogitsLoss(),
	):
		"""

		:param image_size: (int) for initializeing the swin model we need to state the image size explicitly here.
			Note different swin models expect different image sizes using timm
		"""
		super().__init__(
			model_name=model_name,
			learning_rate=learning_rate,
			bins=bins,
			loss_function = loss_function
		)

		self.model = timm.create_model(
			model_name,
			image_size = image_size,
			pretrained=False,
			in_chans=3,
			num_classes = number_of_classes,
			model_drop_rate=model_drop_rate
		)
		self.number_of_classes = number_of_classes
		self.learning_rate = learning_rate


