import enum


class EnumConstant(enum.Enum):
	def __str__(self):
		return self.value


class Datasets(EnumConstant):
	PERSONID = 'person_identification'
	NUM_CLASSES = 246


class DatasetSplit(EnumConstant):
	TRAIN = 'train'
	VAL = 'val'
	TEST = 'test'


class CNNArchitectures(EnumConstant):
	RESNET18 = 'resnet18'
	RESNET34 = 'resnet34'
	RESNET50 = 'resnet50'
	RESNET101 = 'resnet101'


class SiameseArchitectures(EnumConstant):
	SIAMESE = 'SiameseNetwork'


class Metadata(EnumConstant):
	LABEL = 'label'
	IMG_NAME = 'img_name'
	CATEGORY = 'category'
	WIDTH = 'width'
	HEIGHT = 'height'
	SUBSET = 'subset'
	SELFIE_IMG_NAME = 'selfie_img_name'
	SELFIE_WIDTH = 'selfie_width'
	SELFIE_HEIGHT = 'selfie_height'
	IDCARD_IMG_NAME = 'idcard_img_name'
	IDCARD_WIDTH = 'idcard_width'
	IDCARD_HEIGHT = 'idcard_height'


class ImageCategories(EnumConstant):
	SELFIE = 'selfie'
	ID = 'idcard'


class ImageFormats(EnumConstant):
	JPEG = '.jpeg'
	JPG = '.jpg'
	PNG = '.png'
