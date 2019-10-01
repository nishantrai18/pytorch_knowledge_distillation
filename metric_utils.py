import time
import torch

from tensorboardX import SummaryWriter


# Constants
eps = 1e-12


class MetricTracker(object):
    def __init__(self, log_dir, ks=[1, 5], log_frequency=50, starting_step=0):
        self.logger = SummaryWriter(log_dir=log_dir)
        self.step = starting_step
        self.test_step = 0

        self.metrics = {}
        self.num_classes = 100
        self.ks = ks
        self.device = torch.device("cpu")
        # how often to add to logs
        self.log_frequency = log_frequency

        self.losses = None
        self.metrics_tracked = ["top{}_acc".format(k) for k in self.ks]
        self.postfix_losses_list, self.postfix_metrics_list = self.init_postfix_metrics_list()
        self.init_metrics()

    @staticmethod
    def init_postfix_metrics_list():

        postfix_losses, postfix_metrics = dict(), dict()

        postfix_losses["loss"] = "loss"
        postfix_metrics["top1"] = "top1_acc"

        return postfix_losses, postfix_metrics

    def init_metrics(self):
        for metric in self.metrics_tracked:
            self.metrics[metric] = 0

    def init_losses(self, **kwargs):
        self.losses = {k: v.item() for k, v in kwargs.items()}

    def reset_metrics(self):
        for k in self.metrics.keys():
            self.metrics[k] = 0

    def reset_losses(self):
        for k in self.losses.keys():
            self.losses[k] = 0

    def new_epoch(self, phase):
        if self.losses is not None:
            # Means it's not first epoch
            self.log_current_metrics()
        self.reset_epoch_counters()
        self.phase = phase

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def update_losses(self, **kwargs):
        """
        Takes losses as dict as these can be handled automatically
        """
        if self.losses is None:
            self.init_losses(**kwargs)
        else:
            for k, v in kwargs.items():
                self.losses[k] += v.item() * self.batch_size

    def update_metrics(self, outputs, labels):
        """
        Need to calculate all metrics from labels and predictions
        """
        if self.metrics is None:
            self.init_metrics()

        preds = torch.topk(outputs["preds"], k=max(self.ks))[1]
        corrects = preds.eq(labels.view(-1, 1).expand_as(preds))
        correct_top_ks = dict()
        for k in self.ks:
            correct_top_ks[k] = corrects[:, :k].sum().item()

        for metric in self.metrics.keys():
            if metric[:3] == 'top':
                k = int(metric.replace('top', '').split('_')[0])
                self.metrics[metric] += correct_top_ks[k]

        self.inputs_seen_so_far += self.batch_size
        self.step += 1

        if self.step % self.log_frequency == 0:
            self.log_current_metrics()

    def reset_epoch_counters(self):
        if self.losses is not None:
            for k, v in self.losses.items():
                self.losses[k] = 0
        self.init_metrics()
        self.inputs_seen_so_far = 0
        self.start_time = time.time()

    def log_current_metrics(self):
        phase = self.phase.title()

        for name, value in self.losses.items():
            self.logger.add_scalar(phase + "/" + name, value / self.inputs_seen_so_far, self.step)
        for name, value in self.metrics.items():
            self.logger.add_scalar(phase + "/" + name, value / self.inputs_seen_so_far, self.step)

        # Time per 100 steps:
        self.logger.add_scalar(phase + "/Time", (time.time() - self.start_time) * 100 / self.log_frequency, self.step)
        self.start_time = time.time()

        self.reset_losses()
        self.reset_metrics()
        self.inputs_seen_so_far = 0

    def log_test_metrics(self, **kwargs):
        phase = "test".title()

        for name, value in kwargs.items():
            self.logger.add_scalar(phase + "/" + name, value, self.test_step)

        self.test_step += 1

    def fetch_tqdm_postfix_metrics(self):
        postfix_stats = {}
        for k, v in self.postfix_losses_list.items():
            postfix_stats[k] = round(self.losses[v] / (self.inputs_seen_so_far + eps), 3)
        for k, v in self.postfix_metrics_list.items():
            postfix_stats[k] = round(self.metrics[v] / (self.inputs_seen_so_far + eps), 3)
        return postfix_stats

    def update_visuals(self, model):
        """
        Handles updation of visuals generated during training. currently only includes histogram logging.
        Can be used for network activations, etc
        """

        # Log visuals much less frequently compared to scalar metrics
        if self.step % self.log_frequency == 0:
            self.add_beta_histograms(model)

    def add_beta_histograms(self, model):
        """
        Logs histograms to represent the distribution of beta in swish (note: can be generalized)
        """

        phase = self.phase.title()

        beta_val = []
        for n, p in model.named_parameters():
            if p.requires_grad and ("beta" in n):
                beta_val.append(p.item())

        if len(beta_val) > 0:
            self.logger.add_histogram(phase + "/swish_beta", beta_val, self.step)
