import random
from torchvision.transforms import functional as F
import cv2
import numpy as np
import torch
class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ConvertFromInts(object):
    def __call__(self, image, target):
        image = np.array(image, dtype=np.float32)
        return image, target

class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # 使用原图
            None,
            # 最小的IOU，和最大的IOU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, target):
        # 原图的长宽
        _,  height, width = image.shape
        while True:
            # 随机选择一个切割模式
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, target

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # 迭代50次
            for _ in range(50):
                current_image = image.clone()
                # 随机一个长宽
                w = random.uniform(0.1 * width, width)
                h = random.uniform(0.1 * height, height)

                # 判断长宽比在一定范围
                # if h / w < 0.5 or h / w > 2:
                #     continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)

                # 切割的矩形大小，形状是[x1,y1,x2,y2]
                rect = torch.from_numpy(np.array([int(left), int(top), int(left + w), int(top + h)]))

                # 计算切割的矩形和 gt 框的iou大小
                boxes = target["boxes"]
                overlap = jaccard_numpy(boxes, rect)

                # 筛选掉不满足 overlap条件的
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # 从原图中裁剪矩形
                #current_image = current_image[:, rect[1]:rect[3], rect[0]:rect[2]].clone()

                # 所有GT的中心点坐标
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # 这个地方的原理是判断每个GT的中心坐标是否在裁剪框Rect里面，
                # 如果超出了那么下面的mask就全为0，那么mask.any()返回false，
                # 也即是说这次裁剪失败了。
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # 是否有合法的盒子
                if not mask.any():
                    continue

                # 取走中心点落在裁剪区域中的GT
                current_boxes = boxes[mask, :].clone()
                current_boxes = current_boxes.numpy()
                # 从原图中裁剪矩形
                y1 = torch.min(rect[1], boxes[mask, 1]).numpy()
                y2 = torch.max(rect[3], boxes[mask, 3]).numpy()
                x1 = torch.min(rect[0], boxes[mask, 0]).numpy()
                x2 = torch.max(rect[2], boxes[mask, 2]).numpy()
                current_image = current_image[:, int(y1[0]):int(y2[0]), int(x1[0]):int(x2[0])].clone()

                # 取出中心点落在裁剪区域中的GT对应的标签
                labels = target["labels"].clone()
                current_labels = labels[mask]

                # 获取GT box和切割矩形的的最小点xmin、ymin
                #current_boxes[:, :2] = np.minimum(current_boxes[:, :2], rect[:2])
                # 调节坐标系，让boxes的左上角坐标变为切割后的坐标
                #current_boxes[:, :2] -= rect[:2]
                # x1 or xmin
                # torch.max（）返回是个turtle，包含了value，index
                current_boxes[:, 0] = np.maximum(current_boxes[:, 0] - x1, 0)
                # y1 or ymin
                current_boxes[:, 1] = np.maximum(current_boxes[:, 1] - y1, 0)
                # 获取GT box和切割矩形的的最大点xmax、ymax
                # 调节坐标系，让boxes的右上角坐标变为切割后的坐标
                # x2 or xmax
                _,  _height, _width = current_image.shape
                #current_boxes[:, 2], _ = torch.max(_width - (int(x2) - boxes[mask, 2]), 0)
                current_boxes[:, 2] = _width - (current_boxes[:, 2] - x2)
                # y2 or ymax
                #current_boxes[:, 3], _ = torch.max(_height - (int(y2) - boxes[mask, 3]), 0)
                current_boxes[:, 3] = _height - (current_boxes[:, 3] - y2)
                target["labels"] = current_labels
                boxes = torch.from_numpy(current_boxes)
                target["boxes"] = boxes
                # 返回结果
                return current_image, target

class Expand(object):
    # 随机扩充图片
    def __init__(self, mean):
        # 设置一个大于原图尺寸的size，填充指定的像素值mean，然后把原图随机放入这个图片中，实现原图的扩充。
        self.mean = mean

    def __call__(self, image, target):
        if random.randint(2):
            return image, target

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        # 填充mean值
        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        # 放入原图
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image
        # 同样相应的变化boxes的坐标
        boxes = target["boxes"].copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, target

class ConvertColor(object):
    # RGB和HSV颜色空间互转
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError("ConvertColor dose not '{}'.".format("Implemented"))
        return image, target

class RandomSaturation(object):
    # 随机饱和度变化，需要输入图片格式为HSV
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        if random.randint(0, 2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, target

class RandomHue(object):
    # Hue变化需要在 HSV 空间下，改变H的数值，H的取值范围是0-360。
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if random.randint(0, 2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
            return image, target

class RandomContrast(object):
    # 图片的对比度变化，只需要在RGB空间下，乘上一个alpha值。
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, target):
        if random.randint(0, 2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, target

class RandomBrightness(object):
    # 亮度变化只需要在RGB空间下，加上一个delta值
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target):
        if random.randint(0, 2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, target

class SwapChannels(object):
    """图像通道变换
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image

class RandomLightingNoise(object):
    # 图片更换通道，形成的颜色变化
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target

def jaccard_numpy(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    inter = inter[:, 0] * inter[:, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]