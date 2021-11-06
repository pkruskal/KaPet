from typing import Optional
# torch imports
from torch import (
    optim,
    Tensor,
    nn,
)
from torch import cat as concat_tensors
# torch helper imports
import tez
import timm

from utils import RMSELoss

class BaseRegressionModel(tez.Model):
    """
    This base generic model makes use of the torch image models module (timm)
    and adheres to the tez.model framework by adding a loss method and a monitor_metrics
    method both called and output in the forward method.

    Instead of acting as am image classifyer, this model extracts a latent feature space from the images.
    This latent space is used in addition to any other features for a final regression prediction.

    Subclasses should define the image feature network and two regression steps in the initalization.
    """

    def __init__(
            self,
            model_name : str,
            learning_rate : float,
            number_of_latent_image_features : int,
            number_of_additional_features : int,
            number_of_intermediate_regression_variables  : int,
            regression_dropout : float,
            **kwargs
    ) -> nn.Module:
        """
        This is the primary function that will need to be overridden. All code is placeholder code.
        All overriding classes must set
            - self.learning_rate
            - self.model
            - self.dropout
            - self.dense1
            - self.dense2

        Args:
            model_name: (str) associated with from timm's resnet models
                https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
            learning_rate : (float)
        """
        super().__init__()

        self.learning_rate = learning_rate

        # this will need to be overridden to provide the right outputs of <number_of_latent_image_features>
        self.model = timm.create_model(
            model_name
        )

        # set up regression architecture for outputting pawpularity from CNN backbone and other features
        self.dropout = nn.Dropout(regression_dropout)

        # linear module is simple regression y=xA'+b
        # ToDo try adding a nonlinear nn.ReLU activation between dense1 and dense2 i.e. y=reLU(xA'+b)
        number_of_dense1_inputs = number_of_latent_image_features + number_of_additional_features
        self.dense1 = nn.Linear(number_of_dense1_inputs, number_of_intermediate_regression_variables)
        self.dense2 = nn.Linear(number_of_intermediate_regression_variables, 1)

    def loss(
            self,
            estimates : Tensor,
            targets: Optional[Tensor] = None,
            loss_function: nn.Module = RMSELoss()
    ) -> Tensor:
        """
        Calculate the loss between our pawpular estimates and the pawpular targets. If there are no targets then
        we are in evaluation / prediction mode and loss gets defaulted to 0.

        The pawpularity contest uses mean squared error to evaluate loss, but we could try something else.
        Generally it's found that models perform best when trained with the same metric the are evaluated over.

        :param estimates: (Tensor) the pawpular estimates from our model
        :param targets: (Tensor)
        :param loss_function:
        :return:
            loss (Tensor) loss between pawpular estimate and target
        """

        if targets == None:
            loss = Tensor([0])
        else:
            loss = loss_function(estimates.float(), targets.float())

        return loss

    def monitor_metrics(self, prediction, targets=None) -> dict:
        """
        Track the sample variance from a batch if targets are given.

        Args:
            output: pawpularity estimate
            targets: pawpularity targets

        Returns:

        """
        if type(targets) == Tensor:

            #prediction = prediction.cpu().detach().numpy()
            #targets = targets.cpu().detach().numpy()

            sum_squared_error = nn.MSELoss(reduction="sum")(prediction, targets)
            standard_error = (sum_squared_error.sqrt() / (prediction.shape[0] - 2))

            metrics = {
                "sum_squared_error": sum_squared_error,
                "standard_error": standard_error
            }
        else:
            metrics = {

            }

        return metrics

    def forward(self, image, features, targets=None):
        """
        Overrides module.forward

        Following the tez torch trainer framework.
        In this frame work loss and metrics are calculated
        along with model output if targets are assigned.

        The output from the forward step includes image features and the petfinder features along with
        the final pawpularity estimate. These can be used for further downstream processing

        Args:
            image:
            features:
            targets: if assigned loss and metrics will be calculated

        Returns:
            output: tensor dimentions (batch_size x 1+ <number of cnn features> + <number of features>)
                the output for down stream processing includes the final classification which incorperates features
                128 intermediary image features
                all the raw features
            loss:
            metrics:

        """

        # the CNN resnet backbone
        images_latent_space = self.model(image)

        # dense ff network to predict pawpularity
        x = self.dropout(images_latent_space)
        x = concat_tensors([x, features], dim=1)
        x = self.dense1(x)
        pawpularity_estimate = self.dense2(x)

        # return all features with popularity for optional downstream processing over latent space
        output = concat_tensors([pawpularity_estimate, images_latent_space, features], dim=1)

        # calculate the loss if there are targets
        loss = self.loss(pawpularity_estimate, targets=targets)

        # calculate some stats on this batch
        metric_dict = self.monitor_metrics(pawpularity_estimate, targets=targets)

        return output, loss, metric_dict

    def fetch_optimizer(self) -> optim.Optimizer:
        """
        paramaters for the optemizer, this function needs better generalizability
        ToDo fetch_optimizer needs to be able to account for arbitrary optimizers
        ToDo fetch_optimizer needs to be able to follow a learning rate schedule when doing SGD by reading the epoch


        :return:
        """
        lr = self.learning_rate
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer


class BaseReLURegressionModel(tez.Model):
    """
    This base generic model makes use of the torch image models module (timm)
    and adheres to the tez.model framework by adding a loss method and a monitor_metrics
    method both called and output in the forward method.

    Instead of acting as am image classifyer, this model extracts a latent feature space from the images.
    This latent space is used in addition to any other features for a final regression prediction.

    Subclasses should define the image feature network and two regression steps in the initalization.
    """

    def __init__(
            self,
            model_name,
            learning_rate,
            number_of_latent_image_features,
            number_of_additional_features,
            number_of_intermediate_regression_variables,
            regression_dropout,
            **kwargs
    ) -> nn.Module:
        """
        This is the primary function that will need to be overridden. All code is placeholder code.
        All overriding classes must set
            - self.learning_rate
            - self.model
            - self.dropout
            - self.dense1
            - self.dense2

        Args:
            model_name: (str) associated with from timm's resnet models
                https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
            learning_rate : (float)
        """
        super().__init__()

        self.learning_rate = learning_rate

        # this will need to be overridden to provide the right outputs of <number_of_latent_image_features>
        self.model = timm.create_model(
            model_name
        )

        # set up regression architecture for outputting pawpularity from CNN backbone and other features
        self.dropout = nn.Dropout(regression_dropout)

        # linear module is simple regression y=xA'+b
        # ToDo try adding a nonlinear nn.ReLU activation between dense1 and dense2 i.e. y=reLU(xA'+b)
        number_of_dense1_inputs = number_of_latent_image_features + number_of_additional_features
        self.dense1 = nn.Linear(number_of_dense1_inputs, number_of_intermediate_regression_variables)
        self.activation_function = nn.ReLU()
        self.dense2 = nn.Linear(number_of_intermediate_regression_variables, 1)

    def loss(self, estimates, targets=None, loss_function=nn.MSELoss(reduction="sum")):
        # the pawpularity contest uses mean squared error to evaluate loss, but we could try something else

        if targets == None:
            loss = 0
        else:
            loss = loss_function(estimates.float(), targets.float())

        return loss

    def monitor_metrics(self, prediction, targets=None) -> dict:
        """
        Track the sample variance from a batch if targets are given.

        Args:
            output: pawpularity estimate
            targets: pawpularity targets

        Returns:

        """
        if type(targets) == Tensor:

            sum_squared_error = nn.MSELoss(reduction="sum")(prediction, targets)
            standard_error = (sum_squared_error.sqrt() / (prediction.shape[0] - 2))

            metrics = {
                "sum_squared_error": sum_squared_error,
                "standard_error": standard_error
            }
        else:
            metrics = {

            }

        return metrics

    def forward(self, image, features, targets=None):
        """
        Overrides module.forward

        Following the tez torch trainer framework.
        In this frame work loss and metrics are calculated
        along with model output if targets are assigned.

        The output from the forward step includes image features and the petfinder features along with
        the final pawpularity estimate. These can be used for further downstream processing

        Args:
            image:
            features:
            targets: if assigned loss and metrics will be calculated

        Returns:
            output: tensor dimentions (batch_size x 1+ <number of cnn features> + <number of features>)
                the output for down stream processing includes the final classification which incorperates features
                128 intermediary image features
                all the raw features
            loss:
            metrics:

        """

        # the CNN resnet backbone
        images_latent_space = self.model(image)

        # dense ff network to predict pawpularity
        x = self.dropout(images_latent_space)
        x = concat_tensors([x, features], dim=1)
        x = self.dense1(x)
        x = self.activation_function(x)
        pawpularity_estimate = self.dense2(x)

        # return all features with popularity for optional downstream processing over latent space
        output = concat_tensors([pawpularity_estimate, images_latent_space, features], dim=1)

        # calculate the loss if there are targets
        loss = self.loss(pawpularity_estimate, targets=targets)

        # calculate some stats on this batch
        metric_dict = self.monitor_metrics(pawpularity_estimate, targets=targets)

        return output, loss, metric_dict

    def fetch_optimizer(self):
        opt = optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt

    def fetch_scheduler(self):
        """
        Allows adjustment of of learning rate depending on epoch.

        Default is a Cosine Annealing with warm restarts.

        Learning rate is decreased to eta_min over T_0 epochs. then it's reset.
        Then learning rate is decreased to eta_min over T_1 epocs, ... to T_n
        where
        T_n = T_n-1 * T_mult

        :return:
        """
        schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=3, T_mult=2, eta_min=1e-8, last_epoch=-1
        )
        return schedule
