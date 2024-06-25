from models.siamese import SiameseNetwork
from models.resnet import resnet18, resnet34, resnet50, resnet101
from dataset.dataset import PersonIdentificationDataset
from consts.consts import SiameseArchitectures, Datasets


DATASETS = {Datasets.PERSONID: PersonIdentificationDataset}

SIAMESE_ARCHITECTURE = {SiameseArchitectures.SIAMESE: SiameseNetwork}
