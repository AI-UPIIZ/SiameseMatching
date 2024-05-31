import enum


class EnumConstant(enum.Enum):
    def __str__(self):
        return self.value


class DatasetSplit(EnumConstant):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Metadata(EnumConstant):
    SUBSET = "subset"
    

class ImageCategories(EnumConstant):
    SELFIE = "selfie"
    ID = "idcard"
