import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from consts.paths import ModelPaths
from consts.consts import DatasetSplit
from modeling.train.utils.pytorchutils import squeeze_generic, EarlyStopping

class Training:
    def __init__(
        self, input_pipeline, model, loss_function, optimizer, device, early_stop=False
    ):
        self.input_pipeline = input_pipeline
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.initial_epoch = 0
        self.early_stop = early_stop
        self.stopper = EarlyStopping()

        self.epoch = None

        self._epoch_losses = None
        self._epoch_correct = None
        self._epoch_total = None

    def fit(self, train_loader):
        self.model.train()

        self._epoch_losses = 0
        self._epoch_correct = 0
        self._epoch_total = 0

        for _, (selfie_images, idcard_images, labels) in tqdm(
            enumerate(train_loader, 1)
        ):
            selfie_images, idcard_images, labels = (
                selfie_images.to(self.device),
                idcard_images.to(self.device),
                labels.to(self.device),
            )
            self.optimizer.zero_grad()
            (output1, output2), loss = self.forward_to_loss(
                selfie_images, idcard_images, labels
            )

            loss.backward()
            self.optimizer.step()
            self.track_metrics(labels, loss, (output1, output2))

        train_loss = self._epoch_losses / self._epoch_total
        train_acc = self._epoch_correct / self._epoch_total

        return self.model, train_loss, train_acc

    def validate(self, val_loader):
        self.model.eval()

        self._epoch_losses = 0
        self._epoch_correct = 0
        self._epoch_total = 0

        with torch.no_grad():
            for selfie_images, idcard_images, labels in tqdm(val_loader):
                selfie_images, idcard_images, labels = (
                    selfie_images.to(self.device),
                    idcard_images.to(self.device),
                    labels.to(self.device),
                )
                step_output, loss = self.forward_to_loss(
                    selfie_images, idcard_images, labels
                )

                self.track_metrics(labels, loss, step_output)

        val_loss = self._epoch_losses / self._epoch_total
        val_acc = self._epoch_correct / self._epoch_total

        return val_loss, val_acc

    def track_metrics(self, step_labels, loss, step_output):
        step_total = step_labels.size(0)
        step_loss = loss.item()
        self._epoch_losses += step_loss * step_total
        self._epoch_total += step_total

        step_preds = torch.max(step_output.data, 1)[1]
        step_labels = squeeze_generic(step_labels, axes_to_keep=[0])
        step_correct = (step_preds == step_labels).sum().item()

        self._epoch_correct += step_correct

    def forward_to_loss(self, selfie_images, idcard_images, step_labels):
        output1, output2 = self.model(selfie_images, idcard_images)
        step_labels = squeeze_generic(step_labels, axes_to_keep=[0])
        loss = self.loss_function(output1, output2, step_labels)
        return (output1, output2), loss

    def train(self, max_epochs: int):
        train_loss, train_accuracy = [], []
        val_loss, val_accuracy = [], []

        start = time.time()

        logger.info('Starting to fit...')
        for epoch in range(self.initial_epoch, max_epochs):
            model, train_epoch_loss, train_epoch_accuracy = self.fit(
                self.input_pipeline[DatasetSplit.TRAIN]
            )
            val_epoch_loss, val_epoch_accuracy = self.validate(
                self.input_pipeline[DatasetSplit.VAL]
            )

            logger.info(
                f'[Epoch {epoch + 1} of {max_epochs}]: Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}'
            )
            logger.info(
                f'[Epoch {epoch + 1} of {max_epochs}]: Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}'
            )

            train_loss.append(train_epoch_loss)
            train_accuracy.append(train_epoch_accuracy)
            val_loss.append(val_epoch_loss)
            val_accuracy.append(val_epoch_accuracy)

            if self.early_stop:
                self.stopper(val_epoch_loss)
                if self.stopper.early_stop:
                    logger.info(f'Early stopping at epoch [{epoch + 1}]')
                    break

        end = time.time()
        torch.save(self.model.state_dict(), ModelPaths.models_path)
        logger.info(
            f'End training process. Training time: {(end-start)/60:.3f} minutes'
        )

        return model, train_accuracy, val_accuracy, train_loss, val_loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive
