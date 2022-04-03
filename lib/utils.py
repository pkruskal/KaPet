from torch import (
	nn,
	Tensor
)

class RMSELoss(nn.Module):
	"""
	Calculate root mean squared error since it's not built into pytorch

	Note: For optimization purposes MSELoss is the same as RMSELoss,
	BUT Kaggle uses RMSELoss for pawpularity and for human purposes
	it's nice to see the score as listed in the Kaggle Scorebord
	"""

	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()

	def forward(self, yhat : Tensor, y : Tensor):
		"""
		Calculate the root mean square error.

		Args:
			yhat: (Tensor) estimate value
			y: (Tensor) target value

		Returns:

		"""
		return self.mse(yhat, y).sqrt()
