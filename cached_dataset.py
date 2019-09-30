import argparse
import dataset_utils
import glob
import os
import torch
import unittest

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

    def save_results(self, results, mode):
        # Create dirs if doesn't exist
        if not os.path.exists(os.path.join(self.save_dir, mode)):
            os.makedirs(os.path.join(self.save_dir, mode))

        # Save the other files
        for k in results.keys():
            torch.save(results[k], os.path.join(self.save_dir, mode, k + ".pt"))


def dict_collate(data):
    """
    Custom collation function to collate dicts
    """

    # Assuming there's at least one instance in the batch
    add_data_keys = data[0].keys()
    collected_data = {k: [] for k in add_data_keys}

    for i in range(len(list(data))):
        for k in add_data_keys:
            collected_data[k].append(data[i][k])

    for k in add_data_keys:
        collected_data[k] = torch.cat(collected_data[k], 0)

    return collected_data


class CachedKDDataset(torch.utils.data.Dataset):
    """
    Dataset class for efficient KD task
    """

    def __init__(self, mode):
        """
        :param mode: Train or test
        """

        super(CachedKDDataset, self).__init__()

        self.data_dir = os.path.join("../data/cifar100_cached/", mode)
        files = glob.glob(os.path.join(self.data_dir, "*.pt"))

        self.results = {}
        for f in files:
            name = f.replace(".pt", "").split("/")[-1]
            self.results[name] = torch.load(f)

    def __len__(self):
        return self.results["data"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx].unsqueeze(0) for k, v in self.results.items()}


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
    cacher.save_results(results, mode="train")

    results = cacher.perform_pass(test_loader)
    cacher.save_results(results, mode="test")


def fetch_cifar100_efficient_kd_dataloaders(args):
    """
    Returns the efficient knowledge distillation dataloaders. It consists
    of a custom dataloader which returns the image, label and the relevant
    outputs of different models
    """

    loaders = {}
    for mode in ["train", "test"]:
        dataset = CachedKDDataset(mode=mode)
        loaders[mode] = \
            torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=(mode == "train"),
                num_workers=4,
                collate_fn=dict_collate
            )

    return loaders["train"], loaders["test"]


class TestCachedDataloader(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        args = argparse.Namespace(batch_size=64)
        self.train_loader, _ = fetch_cifar100_efficient_kd_dataloaders(args)

    @classmethod
    def tearDownClass(self):
        pass

    def test_cached_dataloader(self):
        """
        Iterates over the data pipeline once
        """

        v = ["data", "target", "model_out_sqnet"]

        for data in self.train_loader:
            b, c, h, w = data[v[0]].shape
            assert data[v[1]].shape == (b, )
            assert data[v[2]].shape == (b, 100)


if __name__ == "__main__":
    unittest.main()
    # create_cached_dataset()