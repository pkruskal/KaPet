from torch import nn
import timm

from models.base_image_regression_model import BaseRegressionModel


class PawpularEfficientNetModel(BaseRegressionModel):
	"""
	This efficientnet generic model makes use of the torch image models module (timm)

	efficientnet is a recent top scoring framework.
	main paper: Rethinking Model Scaling for CNNs - https://arxiv.org/pdf/1905.11946.pdf

	[m for m in timm.list_models() if len(re.findall("^efficient",m)) > 0]
	For more information on these see
	https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py

	* EfficientNet-V2
		- `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
	* EfficientNet (B0-B8, L2 + Tensorflow pretrained AutoAug/RandAug/AdvProp/NoisyStudent weight ports)
		- EfficientNet: Rethinking Model Scaling for CNNs - https://arxiv.org/abs/1905.11946
		- CondConv: Conditionally Parameterized Convolutions for Efficient Inference - https://arxiv.org/abs/1904.04971
		- Adversarial Examples Improve Image Recognition - https://arxiv.org/abs/1911.09665
		- Self-training with Noisy Student improves ImageNet classification - https://arxiv.org/abs/1911.04252
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
