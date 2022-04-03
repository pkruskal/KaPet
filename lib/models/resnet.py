
# torch imports
from torch import (
	nn
)

# torch helper imports
import timm

from lib.models.base_image_regression_model import BaseRegressionModel


class PawpularResnetModel(BaseRegressionModel):
	"""
	This resnet generic model makes use of the torch image models module (timm)

	To see valid model_names try
	[m for m in timm.list_models() if len(re.findall("^resnet",m)) > 0]

	For more information on these see
	https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
	https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnetv2.py

	cross stage partial resnets
	[m for m in timm.list_models() if len(re.findall("^cspresnet",m)) > 0]
	https://github.com/WongKinYiu/CrossStagePartialNetworks

	efficient channel attention resnets
	[m for m in timm.list_models() if len(re.findall("^ecaresnet",m)) > 0]
	https://sotabench.com/paper/eca-net-efficient-channel-attention-for-deep

	"""

	def __init__(
			self,
			model_name: str,
			learning_rate: float,
			number_of_latent_image_features: int = 128,
			number_of_additional_features: int = 12,
			number_of_intermediate_regression_variables: int =64,
			regression_dropout: float = 0.1,
			regression_activation_function: nn.Module = nn.Identity(),
			model_drop_rate=0,
	) -> nn.Module:
		"""
		Args:
			see BaseRegressionModel for all other args
			model_drop_rate: (float between 0 and 1) dropout rate used in the resnet model
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

		# initialize and set the resnet model
		self.model = timm.create_model(
			model_name,
			pretrained=False,
			in_chans=3,
			num_classes=number_of_latent_image_features,
			drop_rate=model_drop_rate
		)

		self.learning_rate = learning_rate

