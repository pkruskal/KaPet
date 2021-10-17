import pandas as pd
import tqdm
import re
from pathlib import Path
import numpy as np
import datetime
import time

import matplotlib.pyplot as plt

from sklearn import model_selection
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch
from PIL import Image

from configuration import load_configuration
from constants import ColumnNames
from transforms import (

)

base_dir = Path().cwd()
test_img_dir = base_dir / 'input/test'
test_data_csv = base_dir / 'input/test.csv'
train_img_dir = base_dir / 'input/train'
train_data_csv = base_dir / 'input/train.csv'
sumbission_csv = base_dir / 'input/sample_submission.csv'
temp_dir = base_dir / 'temp'
store_dir = base_dir / 'working'

config = load_configuration(base_dir)


"""
stopped at epoc 3
early stopping!! best iteration:399 best loss:20.491863250732422 current iteration:2199 current loss:20.636474609375
output tensor([[36.6952]])
output tensor([[36.6952]])
output tensor([[36.6952]])
output tensor([[36.6952]])
output tensor([[36.6952]])
output tensor([[36.6952]])
output tensor([[36.6952]])
output tensor([[36.6952]])
803it [19:32,  1.46s/it]
"""




def add_image_path(data,img_dir):
    data[ColumnNames.image_path.value] = data[ColumnNames.image_name.value].map(lambda x : img_dir / (x + ".jpg"))
    return data

train_data = pd.read_csv(train_data_csv)
test_data = pd.read_csv(test_data_csv)

train_data = add_image_path(train_data,train_img_dir)
test_data = add_image_path(test_data,test_img_dir)

train_data.head()

print(f"total files in image train dir {len(list(train_img_dir.glob('*.*')))}")
print(f"total jpgs in image test dir {len(list(test_img_dir.glob('*.jpg')))}")


class PetfinderImageSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_folder, images_df, transform=None):
        """
        images_df uses columns
        "label" : integer to identify the target,
        "local_path" : str the local path to the image
        data_folder : the path to the data (data_folder/local_path) is the full image
        """

        self.images_df = images_df
        self.transform = transform
        self.data_folder = data_folder

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
        label = self.images_df.iloc[idx][ColumnNames.label.value]
        sample = (image, label)

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


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class VGG(nn.Module):
    VGG_configurations = {
        'conv2': ['CNN_64', 'CNN_64', 'MaxPool', 'View', 'ff_256', 'ff_256', 'ff_10'],
        'conv4': ['CNN_64', 'CNN_64', 'MaxPool', 'CNN_128', 'CNN_128', 'MaxPool', 'View', 'ff_256', 'ff_256', 'ff_10'],
        'conv6': ['CNN_64', 'CNN_64', 'MaxPool', 'CNN_128', 'CNN_128', 'MaxPool', 'CNN_256', 'CNN_256', 'MaxPool',
                  'AvgPool',
                  'View', 'ff_256', 'ff_256', 'ff_10'],
        'VGG19': ['CNN_64', 'CNN_64', 'MaxPool',
                  'CNN_128', 'CNN_128', 'MaxPool',
                  'CNN_256', 'CNN_256', 'CNN_256', 'CNN_256', 'MaxPool',
                  'CNN_512', 'CNN_512', 'CNN_512', 'CNN_512', 'MaxPool',
                  'AvgPool', 'View', 'ff_10'],
        'VGG19_1': ['CNN_64', 'CNN_64', 'MaxPool',
                    'CNN_128', 'CNN_128', 'MaxPool',
                    'CNN_256', 'CNN_256', 'CNN_256', 'CNN_256', 'MaxPool',
                    'CNN_512', 'CNN_512', 'CNN_512', 'CNN_512', 'MaxPool',
                    'AvgPool', 'View', 'ff_1'],
        'VGG19_512_32_1': ['CNN_64', 'CNN_64', 'MaxPool', 'CNN_128', 'CNN_128', 'MaxPool',
                           'CNN_256', 'CNN_256', 'CNN_256', 'CNN_256', 'MaxPool', 'CNN_512',
                           'CNN_512', 'CNN_512', 'CNN_512', 'MaxPool', 'AvgPool', 'View',
                           'ff_512', 'ff_32', 'ff_1', 'View']
    }

    def __init__(self, vgg_name, input_dimention):
        super(VGG, self).__init__()

        self.vgg_name = vgg_name
        self.cfg = self.VGG_configurations[vgg_name]
        self.input_dim = input_dimention
        self.features = self._make_layers(self.cfg)

    def forward(self, x):
        out = self.features(x)
        return out

    def _make_layers(self, layer_type):
        layers = []
        in_channels = 3
        layer_dim = self.input_dim
        for x in layer_type:
            if x == 'MaxPool':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layer_dim = int(layer_dim * 0.5)
            elif x == 'View':
                layers += [View()]
                in_neurons = layer_dim * layer_dim * in_channels
            elif x == 'AvgPool':
                layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
            elif x[:3] == 'CNN':
                out_channels = int(re.split('_', x)[-1])
                convolution_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                batching_layer = nn.BatchNorm2d(out_channels)
                bathing_neuron_out = nn.ReLU(inplace=True)
                layers += [convolution_layer,
                           batching_layer,
                           bathing_neuron_out]
                in_channels = out_channels
            elif x[:2] == 'ff':
                out_neurons = int(re.split('_', x)[-1])
                layers += [nn.Linear(in_neurons, out_neurons)]
                in_neurons = out_neurons
        return nn.Sequential(*layers)


# generate splits
y = train_data[ColumnNames.label.value]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    train_data, y, train_size=0.9, random_state=2019, shuffle=True, stratify=y)
X_train, X_validation, y_train, y_valididation = model_selection.train_test_split(
    X_train, y_train, train_size=0.9, random_state=2019, shuffle=True, stratify=y_train)

# build data sets
dataset_train = PetfinderImageSet(train_img_dir, images_df=X_train, transform=data_transform)
dataset_validation = PetfinderImageSet(train_img_dir, images_df=X_validation, transform=data_transform)
dataset_test = PetfinderImageSet(train_img_dir, images_df=X_test, transform=data_transform)

print(f'train size {len(dataset_train)} batches {len(dataset_train) / config["batchsize"]}')
print(f'validation size {len(dataset_validation)} batches {len(dataset_validation) / config["batchsize"]}')
print(f'internal testing size {len(dataset_test)} batches {len(dataset_test) / config["batchsize"]}')

# build data loaders
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=config["batchsize"]
)
train_valid_loader = torch.utils.data.DataLoader(
    dataset_validation,
    batch_size=config["batchsize"]
)
test_valid_loader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=config["batchsize"]
)


def get_arranged_loaders():
    """
    helper function for the Trainer to pull data loaders as the class is currently structured
    """
    return (train_loader, train_valid_loader, test_valid_loader)


class Trainer():
    """
    Should turn this into an abstract class and specify train_model per simulation while erything else stays genenralized
    """

    def __init__(self, model, model_name, device, data_loader, learning_rate, saveFolder=None):
        self.data_loader_fun = data_loader
        self.device = device
        self.model = model
        self.model_name = model_name

        self.learning_rate = learning_rate

        self.base_folder = saveFolder

        self.evaluation_rate = int(200)  # number of batches between evaluations
        self.early_stoping_check_rate = self.evaluation_rate
        self.early_stoping_threshold = int(self.evaluation_rate * 4)  # threshold in number of batches/iterations
        self.early_stoping_min_run = int(self.evaluation_rate * 10)  # minimum number of batches/iterations to run
        self.check_for_early_stoping = True

        self.train_loss_freq = 50  # frequency of printing the training loss after the first epoch
        self.evaluation_rate_offset = 0  # offsets when testing evaluation < self.evaluation_rate

        # will need to set these by a dataloader at train time, or manually before training
        self.train_loader = None
        self.train_loader_eval = None
        self.test_loader_eval = None

    @staticmethod
    def set_epochs(start_epoch, end_epoch, total_epochs):
        # allow for doing model in chunks
        if type(start_epoch) != int:
            epochs = range(total_epochs)
        elif type(end_epoch) != int:
            epochs = range(start_epoch, total_epochs)
        else:
            epochs = range(start_epoch, end_epoch)
        return epochs

    def data_loader(self):
        self.train_loader, self.train_loader_eval, self.test_loader_eval = self.data_loader_fun()

    def evaluate(self, data_loader, evalutation_type):
        """
        will evaluate the error and accuracy across the whole data set
        :param args:
        :param data_loader:
        :param evalutation_type:
        :return:
        """
        tic = time.time()
        if evalutation_type not in ["Train", "Test", "Validate"]:
            ValueError(f"evalutation_type must be one of Train,Test, or Validate")
        print(f"evaluating {evalutation_type}")
        if not hasattr(self, "track_evaluations"):
            self.track_evaluations = {
                "Train": [],
                "Test": [],
                "Validate": []
            }

        self.model.eval()
        test_loss = 0
        correct = 0
        data_size = 0
        with torch.no_grad():
            for itter, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                sum_squared_error = nn.MSELoss(reduction="sum")
                this_loss = sum_squared_error(torch.flatten(output).float(), target.float())
                """
                _, predicted = output.max(1)
                total = target.size(0)
                correct = predicted.eq(target).sum().item()
                """
                test_loss += this_loss
                data_size += len(target)

        # get the mean squared error
        test_loss /= data_size
        # get the root mean squared error
        test_loss = torch.sqrt(test_loss)
        print(f"test loss {test_loss}")
        print(f"evaluation took {time.time() - tic}")
        self.track_evaluations[evalutation_type].append({
            'batch': self.i_batch,
            'epoch': self.i_epoch,
            'iteration': self.iteration,
            'model_type': self.model_name,
            'loss': test_loss,
            'time': datetime.datetime.now()
        })

        self.model.train()
        return test_loss

    def early_stoping(self):
        """
        will check if there has been no improvment for x itterations on the validation, and if so will stop
        :return:
        """

        validation_losses = [t['loss'] for t in self.track_evaluations["Validate"]]
        min_loss = min(validation_losses)
        min_itteration = self.track_evaluations["Validate"][validation_losses.index(min_loss)]["iteration"]
        if self.iteration - min_itteration > self.early_stoping_threshold:
            mssg_args = [min_itteration, min_loss, self.iteration, validation_losses[-1]]
            print('early stopping!! best iteration:{} best loss:{} current iteration:{} current loss:{}'.format(
                *mssg_args))
            return True
        else:
            return False

    def save_model(self, i_epoch, i_batch):
        model_state = self.model.state_dict()

        save_filename = self.base_folder / "epoch{}_batch_{}_weights.pkl".format(int(i_epoch), i_batch)
        torch.save(model_state, open(save_filename, 'wb'))

    def training_step(self, data, target):
        """
        a single training step based on data batch and target batch
        called by train_model
        :param data:
        :param target:
        :return:
        """
        data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()  # reset the gradients
        # output = model.forward(data)
        output = self.model(data)
        output = torch.flatten(output)  # flatten in the case of regression opposed to category outputs
        mean_squared_error = nn.MSELoss(reduction='mean')
        loss = torch.sqrt(mean_squared_error(output.float(), target.float()))
        loss.backward()  # calculate the errors
        self.optimizer.step()  # update the weights
        return loss, output

    def set_optimizer(self, i_epoch):
        """
        paramaters for the optemizer, this function needs better generalizability
        ToDo set_optimizer needs to be able to follow a learning rate schedule when doing SGD
        ToDo set_optimizer needs to be better at defining the algorithm SGD vs ADAM etc
        :param args:
        :param i_epoch:
        :return:
        """
        if type(self.learning_rate) == float:
            lr = self.learning_rate
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif i_epoch < 80:
            lr = self.learning_rate[0]
            print('setting learning rate as {}'.format(lr))
            # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate[0], momentum=args.momentum)
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate[0], momentum=0.9, weight_decay=5e-4)
        elif i_epoch < 120:
            lr = self.learning_rate[1]
            print('setting learning rate as {}'.format(lr))
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate[1], momentum=0.9, weight_decay=5e-4)
        else:
            lr = self.learning_rate[2]
            print('setting learning rate as {}'.format([lr]))
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate[2], momentum=0.9, weight_decay=5e-4)
        self.optimizer = optimizer

    def update_training_loss(self, loss, output, target, reset_tracker=False):

        rolling_window = 50
        if not hasattr(self, "track_training") or reset_tracker:
            print("initalizing training loss")
            self.track_training = {
                "rolling_index": 0,
                "rolling_training_loss": np.empty((rolling_window)),
                "rolling_dataSize": np.empty((rolling_window)),
                "rolling_dataCorrect": np.empty((rolling_window))
            }
            self.track_training["rolling_training_loss"][:] = np.nan
            self.track_training["status_array"] = []

        # rolling metrics index
        rolling_index = self.track_training["rolling_index"]
        self.track_training["rolling_index"] += 1
        rolling_indx = np.mod(rolling_index, rolling_window)

        # calculate loss
        training_loss = loss.item()

        # put in rolling avg tracker
        self.track_training["rolling_training_loss"][rolling_indx] = training_loss

        # calculate rolling metrics
        rolling_mean_loss = np.nanmean(self.track_training["rolling_training_loss"])
        # print results
        if self.iteration % 10 == 0:
            print(
                f'rolling loss {rolling_mean_loss} | itter {self.iteration} - batch {self.i_batch} - epoch {self.i_epoch}')
        # store results
        self.track_training["status_array"].append({
            'batch': self.i_batch,  # index 1
            'epoch': self.i_epoch,  # index 2
            'iteration': self.iteration,  # index 3
            'model_type': self.model_name,
            'loss': loss.item(),
            'rolling loss': rolling_mean_loss,
            'time': datetime.datetime.now()
        })

    def train_model(self, epochs, start_epoch=None, end_epoch=None):

        if not (self.train_loader and self.train_loader_eval and self.test_loader_eval):
            self.data_loader()
            train_loader = self.train_loader
            train_loader_eval = self.train_loader_eval
            test_loader_eval = self.test_loader_eval

        date = datetime.datetime.now().strftime('%Y_%m_%d')

        if not start_epoch:
            self.iteration = 0
        else:
            self.iteration = start_epoch * len(train_loader)

        # torch.manual_seed(args.seed_dataloader)
        status_tracking = []
        # losses_0pt1 = losses
        running_loss = 0.0

        # allow for training model in chunks at seperate points in time
        epochs = self.set_epochs(start_epoch, end_epoch, epochs)

        for i_epoch in epochs:

            self.i_epoch = i_epoch
            print('working on epoch {}'.format(i_epoch))

            # update optimizers since this can depend on the epoch number
            self.set_optimizer(i_epoch)

            # this is just teling the model it's in training mode so it should track gradients
            self.model.train()

            for i_batch, (data, target) in enumerate(train_loader):
                self.i_batch = i_batch
                loss, output = self.training_step(data, target)

                # reporting progress as a quick check
                if i_epoch == 0:
                    self.update_training_loss(loss, output, target)
                else:
                    if self.iteration % self.train_loss_freq == 0:
                        self.update_training_loss(loss, output, target)

                # evaluations
                if self.iteration % self.evaluation_rate == self.evaluation_rate_offset:
                    print('------')
                    print(f'evaluating model at ittr:{self.iteration} epoch:{i_epoch}')
                    self.evaluate(train_loader_eval, "Validate")
                    self.evaluate(test_loader_eval, "Test")
                    print(' - - - - - -')
                    self.model.train()

                    self.save_model(i_epoch, i_batch)

                if self.iteration % self.early_stoping_check_rate == self.evaluation_rate_offset:
                    if self.check_for_early_stoping and self.iteration > self.early_stoping_min_run:
                        if self.early_stoping():
                            return

                """
                if i_epoch == 0:
                    if i_batch < 50:
                        self.save_model(i_epoch, i_batch)
                    elif i_batch % 5 == 0:
                        # time.sleep(60)
                        self.save_model(i_epoch, i_batch)
                elif i_epoch < 4:
                    if i_batch % 25 == 0:
                        # time.sleep(60)
                        self.save_model(i_epoch, i_batch)
                elif i_epoch < 80:
                    if i_batch % 100 == 0:
                        # time.sleep(60)
                        self.save_model(i_epoch, i_batch)
                elif i_epoch < 120:
                    if i_batch % 250 == 0:
                        # time.sleep(60)
                        self.save_model(i_epoch, i_batch)
                """

                self.iteration += 1

            # save tracking stats
            self.save_model(i_epoch, i_batch)
            pd.DataFrame(self.track_evaluations["Validate"]).to_csv(
                Path(self.base_folder) / f'_status_tracking_{self.model_name}_{date}_Validation.csv')
            pd.DataFrame(self.track_evaluations["Test"]).to_csv(
                Path(self.base_folder) / f'_status_tracking_{self.model_name}_{date}_Test.csv')
            pd.DataFrame(self.track_training["status_array"]).to_csv(
                Path(self.base_folder) / f'_status_tracking_{self.model_name}_{date}_rolling_train.csv')


model = VGG(config["model_configuraton"],config["image_dimention"]).to(config["device"])

save_path = store_dir / f'{config["model_configuraton"]}_VGG_test'
save_path.mkdir(exist_ok = True)

trainer = Trainer(
    model,
    model_name = config["model_configuraton"],
    device = config["device"],
    data_loader = get_arranged_loaders,
    learning_rate = config["learning_rate"],
    saveFolder=save_path
)
trainer.train_loss_freq = 200
trainer.evaluation_rate_offset = 199
trainer.train_model(epochs = config["epocs"])



test_data = pd.read_csv(test_data_csv)
submission = pd.read_csv(sumbission_csv)
sumission_data = test_data.join(submission,rsuffix="_submission")

sumission_data[ColumnNames.label.value] = sumission_data["Id"]
sumission_data = add_image_path(sumission_data,test_img_dir)
sumission_data.head()

trainer.model.eval()

dataset_submission = PetfinderImageSet(test_img_dir, images_df=sumission_data, transform=submission_transform)
submission_loader = torch.utils.data.DataLoader(
    dataset_submission,
    batch_size=1
)

with torch.no_grad():
    for itter, (data, Ids) in enumerate(submission_loader):
        output = trainer.model(
            torch.reshape(data, (-1, 3, 128, 128))
        )
        #output = trainer.model(data.to(trainer.device))
        print(f"output {output}")
        for pawpularity, Id in zip(output.to("cpu"), Ids):
            submission.loc[submission["Id"] == Id, "Pawpularity"] = pawpularity[0].numpy().flatten()[0]
submission.to_csv("submission.csv", index=False)


all_outs = []
with torch.no_grad():
    for itter, (data, Ids) in tqdm.tqdm(enumerate(train_loader)):
        output = trainer.model(data.to(trainer.device))
        all_outs.extend(output.to("cpu"))

plt.figure()
plt.hist(train_data[ColumnNames.label.value])
plt.show()

np_outs = [o[0].numpy().flatten()[0] for o in all_outs]
plt.figure()
plt.hist(np_outs)
plt.show()