import os
import torch

from tqdm import tqdm


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

        :param model: Model to train
        :param train_loader: Train set data loader
        :param test_loader: Test set data loader
        :param optimizer: Optimizer to use for training
        """

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

    def train_step(self, epoch):
        """
        Helper function to schedule training of the specified model. Performs one epoch of
        training
        """

        self.model.train()
        tq = tqdm(self.train_loader, desc="Steps within train epoch {}:".format(epoch))

        for batch_idx, (data, target) in enumerate(tq):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.model.train_loss(output, target)
            loss.backward()
            self.optimizer.step()
            tq.set_postfix({"loss": loss.item()})

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
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                test_loss += self.model.test_loss(output, target).item()
                # get the probable classes
                preds = torch.topk(output, k=5)[1]
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
