from os.path import join
from torchvision import transforms
from dataset import DatasetFromFolderEval, DatasetFromFolder


def transform():
    return transforms.Compose([
        transforms.ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# def transform(fineSize):
#     return transforms.Compose([
#     transforms.Scale(2*fineSize),
#     transforms.RandomCrop(fineSize),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor()])



def get_training_set(data_dir, patch_size, up_factor, data_augmentation):
    data1 = data_dir + 'DIV2K'
    data2 = data_dir + 'Flickr2K'

    return DatasetFromFolder(data1, data2, patch_size, up_factor, data_augmentation,
                             transform=transform())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform())

