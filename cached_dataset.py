import dataset_utils
import os
import torch

import args as ag

from tqdm import tqdm


class DatasetCacher(object):
    """
    Utility class which allows us to go over a dataset, fetch
    different outputs after passing through different models and
    save the result efficiently in records.
    """

    def __init__(self, model_pths, device):
        """
        Initializer for DatasetCacher

        :param model_pths: Map from model_name to model_ckpt_pth
        """
        self.device = device
        self.models = self.load_models(model_pths)

        self.save_dir = "../data/cifar100_cached/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load_models(self, model_pths):
        models = {}
        for k, pth in model_pths.items():
            model = torch.load(pth)
            models['model_out_' + k] = model
        return models

    def perform_pass(self, loader):
        tq = tqdm(loader, desc="Steps within dataset:")

        results = {x: list() for x in self.models.keys()}
        results["data"] = list()
        results["target"] = list()

        for batch_idx, (data, target) in enumerate(tq):
            data, target = data.to(self.device), target.to(self.device)
            results["data"].append(data.detach())
            results["target"].append(target.detach())
            # Compute result for each model
            for k, model in self.models.items():
                results[k].append(model(data)["outs"].detach())

        for k in results.keys():
            results[k] = torch.cat(results[k])

        return results

    def save_results(self, results):
        for k in results.keys():
            torch.save(results[k], os.path.join(self.save_dir, k + ".pt"))


class CachedKDDataset(torch.utils.data.Dataset):
    """
    Dataset class for efficient KD task
    """

    def __init__(self):
        super(CachedKDDataset, self).__init__()

    def __getitem__(self, idx):
        pass


def create_cached_dataset():

    parser = ag.get_args_parser_for_cached_dataset()
    args = parser.parse_args()

    torch.manual_seed(42)

    train_loader, test_loader = dataset_utils.fetch_cifar100_dataloaders(args, num_workers=0)

    device = torch.device("cpu")
    model_pths = {
        "sqnet": "../model_ckpt/sqnet_with_aug/cifar100_3.pt"
    }

    cacher = DatasetCacher(model_pths, device)

    results = cacher.perform_pass(train_loader)

    cacher.save_results(results)


if __name__ == '__main__':
    create_cached_dataset()
