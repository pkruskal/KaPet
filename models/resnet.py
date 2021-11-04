
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

	there is a rich history of resnet models

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

