import torch

from tqdm import tqdm


def train_model(model, device, train_loader, optimizer, loss_criterion, epoch):
    """
    Helper function to schedule training of the specified model
        :param model: Model to train
        :param train_loader: Train set data loader
        :param optimizer: Optimizer to use for training
        :param loss_criterion: Loss criteria to use for training
    """

    model.train()
    tq = tqdm(train_loader, desc="Steps within train epoch {}:".format(epoch))

    for batch_idx, (data, target) in enumerate(tq):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, target)
        loss.backward()
        optimizer.step()
        tq.set_postfix({"loss": loss.item()})


def test_model(model, device, test_loader, loss_criterion, ks=[1, 5]):
    """
    Helper function to schedule training of the specified model
        :param model: Model to train
        :param test_loader: Test set data loader
        :param loss_criterion: Loss criteria to use for training
        :param ks: Ks for which to compute TopK accuracy
    """

    model.eval()
    test_loss = 0
    correct_top_ks = {k: 0 for k in ks}

    with torch.no_grad():
        tq = tqdm(test_loader, desc="Steps within test:")
        for data, target in tq:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += loss_criterion(output, target).item()
            # get the probable classes
            preds = torch.topk(output, k=5)[1]
            corrects = preds.eq(target.view(-1, 1).expand_as(preds))
            for k in ks:
                correct_top_ks[k] += corrects[:, :k].sum().item()

    test_loss /= len(test_loader.dataset)

    # Print relevant stats
    print("Test set: Average loss: {:.4f}".format(test_loss))
    for k in ks:
        print("Accuracy Top {}: {}/{} ({:.1f}%)".format(
            k, correct_top_ks[k], len(test_loader.dataset),
            100. * correct_top_ks[k] / len(test_loader.dataset))
        )
