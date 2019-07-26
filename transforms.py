"""Functions for data augmentation and related preprocessing."""

import random
import numpy as np
import cv2


def color_aug(img, random_h=36, random_l=50, random_s=50):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(float)

    h = (random.random() * 2 - 1.0) * random_h
    l = (random.random() * 2 - 1.0) * random_l
    s = (random.random() * 2 - 1.0) * random_s

    img[..., 0] += h
    img[..., 0] = np.minimum(img[..., 0], 180)

    img[..., 1] += l
    img[..., 1] = np.minimum(img[..., 1], 255)

    img[..., 2] += s
    img[..., 2] = np.minimum(img[..., 2], 255)

    img = np.maximum(img, 0)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HLS2BGR)
    return img


class GroupCenterCrop(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size):
        self._size = size

    def __call__(self, img_group):
        h, w, _ = img_group[0].shape
        hs = (h - self._size) // 2
        ws = (w - self._size) // 2
        return [img[hs:hs+self._size, ws:ws+self._size] for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_mv=False):
        self._is_mv = is_mv

    def __call__(self, img_group, is_mv=False):
        if random.random() < 0.5:
            # input_frame:height * width * 3(RGB_channels)
            # img[:, ::-1, :]  flip horizontally
            ret = [img[:, ::-1, :].astype(np.int32) for img in img_group]
            if self._is_mv:
                for i in range(len(ret)):
                    ret[i] -= 128  # invert flow pixel values when flipping
                    ret[i][..., 0] *= (-1)
                    ret[i] += 128
            return ret
        else:
            return img_group


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size):
        self._size = (size, size)

    def __call__(self, img_group):
        if img_group[0].shape[2] == 3: # RGB
            return [cv2.resize(img, self._size, cv2.INTER_LINEAR) for img in img_group]
        elif img_group[0].shape[2] == 2:
            return [resize_mv(img, self._size, cv2.INTER_LINEAR) for img in img_group]
        else:
            assert False


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, is_mv=False):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self._is_mv = is_mv

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h, _ = img_group[0].shape
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()

        for o_w, o_h in offsets:
            for img in img_group:

                # crop a (crop_size*crop_size) patch from image set
                crop = img[o_w:o_w+crop_w, o_h:o_h+crop_h]
                oversample_group.append(crop)
                # flip horizontally
                flip_crop = crop[:, ::-1, :].astype(np.int32)

                if self._is_mv:
                    assert flip_crop.shape[2] == 2, flip_crop.shape
                    flip_crop -= 128
                    flip_crop[..., 0] *= (-1)
                    flip_crop += 128
                oversample_group.append(flip_crop)

        return oversample_group

#          ret_img_group = [resize_mv(img, (self.input_size[0], self.input_size[1]), cv2.INTER_LINEAR)
#                              for img in crop_img_group]

# mv image:
def resize_mv(img, shape, interpolation):
    return np.stack([cv2.resize(img[..., i], shape, interpolation)
                     for i in range(2)], axis=2)

# stack()函数的原型是numpy.stack(arrays, axis=0)，即将一堆数组的数据按照指定的维度进行堆叠。
# axis=2, 在第三维度进行堆叠。即若原数组为3*4*5，两个数组堆叠，结果为 3*4*2*5

# stack: ([img0,img1],axis=2)
# -> height * width * 2 * channels

#  cv::INTER_LINEAR = 1

# img: h * w * 2(channels, R(x) & G(y))
# stack: np.stack(([h1,w1,c1,0], [h2,w2,c2,1]),axis=2)

# for i in range(2): resize images of x-mv and y-mv respectively
# after resizing two images(x-channel and y-channel), stack them together
# numpy.stack
# numpy.stack(arrays, axis=0, out=None)[source]
# Join a sequence of arrays along a new axis.
# The axis parameter specifies the index of the new axis in the dimensions of the result.
# For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last dimension.
#
# Parameters:
            # arrays : sequence of array_like           Each array must have the same shape.
            # axis : int, optional                      The axis in the result array along which the input arrays are stacked.
            # out : ndarray, optional                   If provided, the destination to place the result. The shape must be correct, matching that of what stack would have returned if no out argument were specified.
# Returns:
            #  stacked : ndarray                        The stacked array has one more dimension than the input arrays.

class GroupMultiScaleCrop(object):
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=False, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]

    def __call__(self, img_group):

        im_size = img_group[0].shape # a tuple of number of rows, columns and channels
                                    #  If image is grayscale, tuple returned contains only number of rows and columns.
        # img.shape returns (Height, Width, Number of Channels)
        # height = img.shape[0]
        # width = img.shape[1]
        # number_of_channels = img.shape[2]

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)

        # crop images
        crop_img_group = [img[offset_w:offset_w + crop_w, offset_h:offset_h + crop_h] for img in img_group]

        # resize images
        if crop_img_group[0].shape[2] == 3: # RGB
                            # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
                            # Syntax: cv2.resize(image,(width,height))
            ret_img_group = [cv2.resize(img, (self.input_size[0], self.input_size[1]),
                                        cv2.INTER_LINEAR)
                             for img in crop_img_group]
        elif crop_img_group[0].shape[2] == 2: # mv
            # In a motion vector image
            # we assume the x motion is stored in the red channel
            # and the y value is store in the green channel.
            ret_img_group = [resize_mv(img, (self.input_size[0], self.input_size[1]), cv2.INTER_LINEAR)
                             for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        # compute crop size
        image_w, image_h = im_size[0], im_size[1] # width & height of input images(frames)

        base_size = min(image_w, image_h)  # use the length of the shorter edge
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort: # default max_distort = 1
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

####################
