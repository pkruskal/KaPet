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

    Instead of acting as am image classifier, this model extracts a latent feature space from the images.
    This latent space is used in addition to any other features for a final regression prediction .
    The final regression consists of two regression steps (y=xA'+b)
    with the potential of an activation function between them reducint the output to a single value and
    including any other metadata to the image representation.

    Subclasses should define the image feature network and two regression steps in the initalization.
    """

    def __init__(
            self,
            model_name : str,
            learning_rate : float,
            number_of_latent_image_features : int,
            number_of_additional_features : int,
            number_of_intermediate_regression_variables : int,
            regression_dropout : float,
            regression_activation_function : nn.Module = nn.Identity(),
            **kwargs
    ) -> nn.Module:

        """
        This is the primary function that will need to be overridden. All code is placeholder code.
        All overriding classes must set

        :param model_name: (str) associated with from timm's models
        :param learning_rate: (float) initial learning rate for the model
        :param number_of_latent_image_features: (int) the number of outputs from the image backbone model
        :param number_of_additional_features: (int) the number of features that will be added in the metadata
            This must match the number of features output by the dataloader.
        :param number_of_intermediate_regression_variables: (int) the number of features after the first regression
        :param regression_dropout: (float) dropout between the the image latent space and the regression output
        :param regression_activation_function: (nn.Module) default is no activation function.
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
        number_of_dense1_inputs = number_of_latent_image_features + number_of_additional_features
        self.dense1 = nn.Linear(number_of_dense1_inputs, number_of_intermediate_regression_variables)
        self.activation_function = regression_activation_function
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
        :param targets: (Tensor) labeled data
        :param loss_function: (nn.Module) for calculating loss
        :return:
            loss (Tensor) loss between pawpular estimate and target
        """

        if targets == None:
            loss = Tensor([0])
        else:
            loss = loss_function(estimates.float(), targets.float())

        return loss

    def monitor_metrics(self, prediction: Tensor, targets: Optional[Tensor]=None) -> dict:
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

    def forward(
            self, image:Tensor, features:Tensor, targets:Optional[Tensor]=None
    ) -> (Tensor,Tensor,dict):
        """
        Overrides module.forward

        Following the tez torch trainer framework.
        In this frame work loss and metrics are calculated
        along with model output if targets are assigned.

        The output from the forward step includes image features and the petfinder features along with
        the final pawpularity estimate. These can be used for further downstream processing

        Args:
            image: (Tensor) image output from the dataloader
            features: (Tensor) metadata features output from the dataloader
            targets: if we have labels these should match the the images provided

        Returns:
            output: tensor dimensions (batch_size x 1+ <number of image features> + <number of features>)
                the output for down stream processing includes the final classification which incorporates features
                intermediary image features
                all other metadata features
            loss: (Tensor) the loss between targets and estimates if targets are given, else this is 0
            metrics: (Dict) a dictionary of values being tracked if targets are given

        """

        # the CNN resnet backbone
        images_latent_space = self.model(image)

        # dense ff network to predict pawpularity
        x = self.dropout(images_latent_space)
        x = concat_tensors([x, features], dim=1)
        x = self.dense1(x)
        x = self.activation_function(x) # allow for additional activation functions here
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
        This fetches the optimizer for the model. Default is the Adam optimizer.

        :return:
            opt: (optim.Optimizer) an optimizer for gradient accent
        """
        opt = optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt

    def fetch_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """
        Allows adjustment of of learning rate depending on epoch.
        A good reference for this is here
        https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling

        Default is a Cosine Annealing with warm restarts.

        Learning rate is decreased to eta_min over T_0 epochs. then it's reset.
        Then learning rate is decreased to eta_min over T_1 epocs, ... to T_n
        where
        T_n = T_n-1 * T_mult

        :return:
            schedule: (optim.lr_scheduler._LRScheduler)
        """
        schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=3, T_mult=2, eta_min=1e-8, last_epoch=-1
        )
        return schedule

