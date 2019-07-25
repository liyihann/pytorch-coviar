"""Run testing given a trained model."""

import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision

from dataset import CoviarDataSet
from model import Model
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--data-name', type=str, choices=['ucf101', 'hmdb51'])
parser.add_argument('--representation', type=str, choices=['iframe', 'residual', 'mv'])
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')
parser.add_argument('--data-root', type=str)
parser.add_argument('--test-list', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', type=str)
parser.add_argument('--save-scores', type=str, default=None)

parser.add_argument('--test_segments', type=int, default=25)
# --test-segments specifies how many segments to sample in a "TSN (temporal segment network)".
# Many recent papers calculate the video-level prediction as the average of 25 frame-level predictions
# (the 25 frames are sampled uniformly in a video).
# So we follow and set the default as 25 here.

parser.add_argument('--test-crops', type=int, default=10)
# --test-crops specifies how many crops per segment.
# The value should be 1 or 10.
# 1 means using only one center crop.
# 10 means using 5 crops for both (horizontal) flips.

parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of workers for data loader.')
parser.add_argument('--gpus', nargs='+', type=int, default=None)







args = parser.parse_args()

if args.data_name == 'ucf101':
    num_class = 101
elif args.data_name == 'hmdb51':
    num_class = 51
else:
    raise ValueError('Unknown dataset '+args.data_name)

def main():
    # load trained model

    '''
    @Param
    num_class: total number of classes
    num_segments: number of TSN segments, test default = 25
    representation: iframe, mv, residual
    base_model: base architecture
    '''
    net = Model(num_class, args.test_segments, args.representation,
                base_model=args.arch)

    # checkpoint trained model ? (not best model
    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)


    # -----------------------
    # CLASS torchvision.transforms.Compose(transforms)[SOURCE]
    # Composes several transforms together.
    # Parameters: transforms (list of Transform objects) – list of transforms to compose.
    # -----------------------

    # -----------------------
    # TSN:
    # if args.test_crops == 1:
    #     cropping = torchvision.transforms.Compose([
    #         GroupScale(net.scale_size),
    #         GroupCenterCrop(net.input_size),
    #     ])
    # -----------------------
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.crop_size),
        ])

    # ??? what's difference between net.input_size and net.crop_size?
    # -----------------------
    # TSN:
    # elif args.test_crops == 10:
    #     cropping = torchvision.transforms.Compose([
    #         GroupOverSample(net.input_size, net.scale_size)
    #     ])
    # -----------------------

    # is_mv=(args.representation == 'mv') seems quite important
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.crop_size, net.scale_size, is_mv=(args.representation == 'mv'))
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported, but got {}.".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.test_segments,
            representation=args.representation,
            transform=cropping, # seems important to stacking

            # Stack(roll=args.arch == 'BNInception') seems important

            # -----------------------
            # TSN:
            # transform=torchvision.transforms.Compose([
            #     cropping,
            #     Stack(roll=args.arch == 'BNInception'),       # this line seems important
            #     ToTorchFormatTensor(div=args.arch != 'BNInception'),
            #     GroupNormalize(net.input_mean, net.input_std),
            # ])),
            # -----------------------


            is_train=False,
            accumulate=(not args.no_accumulation),
            ),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()

    data_gen = enumerate(data_loader)

    total_num = len(data_loader.dataset)
    output = []

    def forward_video(data):
        input_var = torch.autograd.Variable(data, volatile=True)
        scores = net(input_var)
        scores = scores.view((-1, args.test_segments * args.test_crops) + scores.size()[1:])
        scores = torch.mean(scores, dim=1)
        return scores.data.cpu().numpy().copy()


    proc_start_time = time.time()


    for i, (data, label) in data_gen:
        video_scores = forward_video(data)
        output.append((video_scores, label[0]))
        cnt_time = time.time() - proc_start_time
        if (i + 1) % 100 == 0:
            print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                            total_num,
                                                                            float(cnt_time) / (i+1)))

    video_pred = [np.argmax(x[0]) for x in output]
    video_labels = [x[1] for x in output]

    print('Accuracy {:.02f}% ({})'.format(
        float(np.sum(np.array(video_pred) == np.array(video_labels))) / len(video_pred) * 100.0,
        len(video_pred)))


    if args.save_scores is not None:

        name_list = [x.strip().split()[0] for x in open(args.test_list)]
        order_dict = {e:i for i, e in enumerate(sorted(name_list))}

        reorder_output = [None] * len(output)
        reorder_label = [None] * len(output)
        reorder_name = [None] * len(output)

        for i in range(len(output)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = output[i]
            reorder_label[idx] = video_labels[i]
            reorder_name[idx] = name_list[i]

        np.savez(args.save_scores, scores=reorder_output, labels=reorder_label, names=reorder_name)


if __name__ == '__main__':
    main()
