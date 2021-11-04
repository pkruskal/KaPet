import timm

from base_image_regression_model import BaseRegressionModel

class PawpularSwinModel(BaseRegressionModel):
	"""
	This Swin model makes use of the torch image models module (timm)

	For more info on the Swin image recognition framework
	https://arxiv.org/pdf/2103.14030.pdf

	To see valid model_names try
	[m for m in timm.list_models() if "swin" in m]

	For more information on these configurations see
	https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
	https://github.com/microsoft/Swin-Transformer/tree/main/configs




	See BaseRegressionModel for documentation on this class

	"""

	def __init__(
			self,
			model_name,
			learning_rate,
			number_of_latent_image_features = 128,
			number_of_additional_features = 12,
			number_of_intermediate_regression_variables =64,
			regression_dropout = 0.1,
			image_size = 128
	):
		super().__init__(
			model_name,
			learning_rate,
			number_of_latent_image_features,
			number_of_additional_features,
			number_of_intermediate_regression_variables,
			regression_dropout
		)

		self.model = timm.create_model(
			model_name,
			image_size = image_size,
			pretrained=False,
			in_chans=3,
			num_classes = number_of_latent_image_features
		)

