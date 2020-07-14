"""
test_post_hoc_lib.py

Test the post hoc library
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

from celeb_race import CelebRace, unambiguous
from post_hoc_lib import DebiasModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


descriptions = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
                'Young', 'White', 'Black', 'Asian', 'Index', 'Female']


def load_celeba(input_size=224, num_workers=2, trainsize=10000, testsize=1000, batch_size=32, transform_type='normalize'):
    """Load CelebA dataset"""

    if transform_type == 'normalize':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.ToTensor()

    trainset = CelebRace(root='./data', download=True, split='train', transform=transform)
    testset = CelebRace(root='./data', download=True, split='test', transform=transform)

    # return only the images which were predicted white, black, or asian by >70%.
    trainset = unambiguous(trainset, split='train')
    testset = unambiguous(testset, split='test')

    if trainsize >= 0:
        # cut down the training set
        trainset, _ = torch.utils.data.random_split(trainset, [trainsize, len(trainset) - trainsize])
    trainset, valset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.7), int(len(trainset)*0.3)])
    if testsize >= 0:
        testset, _ = torch.utils.data.random_split(testset, [testsize, len(testset) - testsize])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader


def get_resnet_model():
    """Get Pretrained resnet model"""
    resnet18 = models.resnet18(pretrained=True)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 2)
    return resnet18


class CustomModel(DebiasModel):
    """Custom Model based built on Debias Model"""

    def __init__(self):
        """Initialize"""
        super().__init__()
        self.lam = 0.75
        self.bias_measure = 'aod'
        self.__protected_index = descriptions.index('Black')
        self.__prediction_index = descriptions.index('Smiling')
        self.model = get_resnet_model()
        self.model.to(device)
        self.model.load_state_dict(torch.load('bs_checkpoint.pt')['model_state_dict'])
        self.loaders = load_celeba()

    @property
    def protected_index(self):
        """Index for protected attribute"""
        return self.__protected_index

    @property
    def prediction_index(self):
        """Index for prediction attribute"""
        return self.__prediction_index

    def get_valloader(self):
        """get the valloader"""
        return self.loaders[1]

    def get_testloader(self):
        """get the testloader"""
        return self.loaders[2]

    def get_model(self):
        """load model"""
        return self.model

    def get_last_layer_name(self):
        """Last fully connected layer of network."""
        return 'fc'


def main():
    """Main Function"""
    # CustomModel is a subclass of DebiasModel.
    custom_model = CustomModel()
    # This returns a dictionary containing bias statistics for the original model.
    # If verbose is True, then it prints out the bias statistics.
    orig_data = custom_model.evaluate_original(verbose=True)

    # This runs the random debiasing algorithm on the model and returns
    # the random debiased model and the random threshold that will maximize the objective.
    rand_model, rand_thresh = custom_model.random_debias_model()
    # This returns a dictionary containing bias statistics for the random debiased model.
    # If verbose is True, then it prints out the bias statistics.
    rand_data = custom_model.evaluate_random_debiased(verbose=True)

    # This runs the adversarial debiasing algorithm on the model and returns
    # the adversarial debiased model and the adversarial threshold that will maximize the objective.
    adv_model, adv_thresh = custom_model.adversarial_debias_model()
    # This returns a dictionary containing bias statistics for the adversarial debiased model.
    # If verbose is True, then it prints out the bias statistics.
    adv_data = custom_model.evaluate_adversarial_debiased(verbose=True)


if __name__ == "__main__":
    main()
