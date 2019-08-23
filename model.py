"""Model definition."""

from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision

# MV_STACK_SIZE = 5

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation,
                 base_model='resnet152',mv_stack_size = 1):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        self.mv_stack_size = mv_stack_size

        print(("""
Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)


    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)

            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _prepare_tsn(self, num_class):

        feature_dim = getattr(self.base_model, 'fc').in_features
        setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))
        if self._representation == 'mv':
            setattr(self.base_model, 'conv1',
                    # Conv2d:
                    #     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    #                  padding=0, dilation=1, groups=1,
                    #                  bias=True, padding_mode='zeros'):

                    # concatenate mv stack on 3-th axis(x/y info)
                    # and change in_channels here from 2 to 2*MV_STACK_SIZE

            # -----------------------------ORIGINAL_CODE_START-----------------------------
            #         nn.Conv2d(2, 64,
            #                   kernel_size=(7, 7),
            #                   stride=(2, 2),
            #                   padding=(3, 3),
            #                   bias=False))
            # self.data_bn = nn.BatchNorm2d(2)
            # -----------------------------ORIGINAL_CODE_END-------------------------------
            # -----------------------------MODIFIED_CODE_START-----------------------------
                    nn.Conv2d(2*self.mv_stack_size, 64,
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False))
            self.data_bn = nn.BatchNorm2d(2*self.mv_stack_size)
            # -----------------------------MODIFIED_CODE_END-------------------------------
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)



    def forward(self, input):
        input = input.view((-1, ) + input.size()[-3:])
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)

        base_out = self.base_model(input)
        return base_out

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))])

    # seems the same as TSN
    # -----------------------
    # TSN:
    # elif self.modality == 'Flow':
    # return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
    #                                        GroupRandomHorizontalFlip(is_flow=True)])
    # -----------------------