import torch


class LossCalculator(object):
    def __init__(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_seq = []

    def calc_loss(self, output, target):
        loss = self.criterion(output, target)
        self.loss_seq.append(loss.item())
        return loss

    def get_loss_log(self):
        return sum(self.loss_seq) / len(self.loss_seq)
