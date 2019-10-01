import os
import torch

import metric_utils as mu

from kd_module import ModelWrapper
from tqdm import tqdm


def move_dict_to_device(data_dict, device):
    """
    Assumes that the values in the data_dict have .to(device)
    defined on them
    """
    for k in data_dict.keys():
        data_dict[k] = data_dict[k].to(device)
    return data_dict


def move_to_device(data, device):
    if type(data) == dict:
        return move_dict_to_device(data, device)
    else:
        return data.to(device)


class ModelTrainer(object):

    def __init__(
            self,
            model,
            device,
            train_loader,
            test_loader,
            optimizer
    ):
        """
        Init class for Model Trainer

        :param model: Model to train - should be instance of ModelWrapper
        :param train_loader: Train set data loader
        :param test_loader: Test set data loader
        :param optimizer: Optimizer to use for training
        """

        assert isinstance(model, ModelWrapper)

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.metric_logger = mu.MetricTracker(self.model.log_dir)

        print("Training model {}".format(self.model.name))

    def train_step(self, epoch):
        """
        Helper function to schedule training of the specified model. Performs one epoch of
        training
        """

        self.metric_logger.new_epoch("train")

        self.model.train()
        tq = tqdm(self.train_loader, desc="Steps within train epoch {}:".format(epoch))

        for batch_idx, (data, target) in enumerate(tq):
            data, target = move_to_device(data, self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss_dict = self.model.train_loss(output, target)

            # Batch size can be different for last step
            self.metric_logger.update_batch_size(data.shape[0])
            self.metric_logger.update_losses(**loss_dict)
            self.metric_logger.update_metrics(output, target)
            self.metric_logger.update_visuals(self.model)

            loss_dict["loss"].backward()
            self.optimizer.step()

            tq.set_postfix(**self.metric_logger.fetch_tqdm_postfix_metrics())

    def test_step(self, ks=[1, 5]):
        """
        Helper function to schedule training of the specified model. Performs one test step

        :param ks: Ks for which to compute TopK accuracy
        """

        self.model.eval()
        test_loss = 0
        correct_top_ks = {k: 0 for k in ks}

        with torch.no_grad():
            tq = tqdm(self.test_loader, desc="Steps within test:")
            for data, target in tq:
                data, target = move_to_device(data, self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                test_loss += self.model.test_loss(output, target)["loss"].item()
                # get the probable classes
                preds = torch.topk(output["preds"], k=max(ks))[1]
                corrects = preds.eq(target.view(-1, 1).expand_as(preds))
                for k in ks:
                    correct_top_ks[k] += corrects[:, :k].sum().item()

        test_loss /= len(self.test_loader.dataset)

        # Print relevant stats
        print("Test set: Average loss: {:.4f}".format(test_loss))
        for k in ks:
            print("Accuracy Top {}: {}/{} ({:.1f}%)".format(
                k, correct_top_ks[k], len(self.test_loader.dataset),
                100. * correct_top_ks[k] / len(self.test_loader.dataset))
            )

    def save_model(self, model_save_dir, epoch):
        torch.save(
            self.model,
            os.path.join(model_save_dir, "cifar100_" + str(epoch) + ".pt")
        )
