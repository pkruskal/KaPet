
# torch imports
from torch import (
	nn
)

# torch helper imports
import timm

from models.base_image_regression_model import BaseRegressionModel

def check_model_name(model_name):
	if model_name not in [

	]:
		raise ValueError

class PawpularResnetModel(BaseRegressionModel):
	"""
	This resnet generic model makes use of the torch image models module (timm)
	and adheres to the tez.model framework by adding
	a loss method and a monitor_metrics method both called and output in the forward method.

	Instead of acting as a classifyer, this resnet model extracts a latent feature space from the images.
	This latent space in addition to any other features can then be used for a final regression model.

	"""

	def __init__(
			self,
			model_name,
			learning_rate,
			number_of_latent_image_features = 128,
			number_of_additional_features = 12,
			number_of_intermediate_regression_variables =64,
			regression_dropout = 0.1,
			model_drop_rate=0
	) -> nn.Module:
		"""
		Args:
			model_name: (str) associated with from timm's resnet models
				https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
			learning_rate : (float)
			number_of_resnet_features: (int) how many latent features will be trained by the resnet cnn backbone
			number_of_additional_features: (int) number of features being added to the resnet cnn backbone
			number_of_intermediate_dense_variables (int) number of latent variables to use for the
				non CNN feed forward output model
			drop_rate: (float) 0-1 default 0 the dropout rate within the resnet architecture
		"""
		super().__init__(
			model_name,
			learning_rate,
			number_of_latent_image_features,
			number_of_additional_features,
			number_of_intermediate_regression_variables,
			regression_dropout
		)

		# initalize a resnet model
		self.model = timm.create_model(
			model_name,
			pretrained=False,
			in_chans=3,
			num_classes=number_of_latent_image_features,
			drop_rate=model_drop_rate
		)

		self.learning_rate = learning_rate

