# @author AmythistHe
# @version 1.0
# @description
# @create 2021/3/24 11:14
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.transforms.py_transforms as py_transforms
from mindspore import Tensor
from mindspore.ops import composite as C
import mindspore.ops as ops
import numpy as np
import mindspore


# def norm(img, vgg=False):
#     if vgg:
#         # normalize for pre-trained vgg model
#         # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
#         transform = py_vision.Normalize(mean=[0.485, 0.456, 0.406],
#                                         std=[0.229, 0.224, 0.225])
#     else:
#         # normalize [-1, 1]
#         transform = py_vision.Normalize(mean=[0.5, 0.5, 0.5],
#                                         std=[0.5, 0.5, 0.5])
#     # return transform(Tensor(img, dtype=mindspore.float32))
#     return transform(img)

def norm(img, vgg=False):
    if vgg:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    transform = py_vision.Normalize(mean, std)
    return transform(img)


# def norm_vgg():
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     return py_transforms.Compose([
#         py_vision.Normalize(mean, std)
#     ])

# def norm(data, vgg=False):
#     if vgg:
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#     else:
#         mean = np.array([0.5, 0.5, 0.5])
#         std = np.array([0.5, 0.5, 0.5])
#     mean = mean.reshape(3, 1, 1)
#     std = std.reshape(3, 1, 1)
#     trans = (data - mean) / std
#     return trans


def denorm(data, vgg=False):
    if vgg:
        mean = np.array([-2.118, -2.036, -1.804])
        std = np.array([4.367, 4.464, 4.444])
        mean = mean.reshape(3, 1, 1)
        std = std.reshape(3, 1, 1)
        trans = (data - mean) / std
        return Tensor(trans, dtype=mindspore.float32)
    else:
        out = Tensor((data + 1) / 2, dtype=mindspore.float32)
        return C.clip_by_value(out, 0, 1)


def gram_matrix(input):
    a, b, c, d = input.shape  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    matmul = ops.MatMul()
    transpose = ops.Transpose()
    div = ops.Div()
    perm = (1, 0)
    G = matmul(features, transpose(features, perm))  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return div(G, (a * b * c * d))


# def denorm(img, vgg=False):
#     if vgg:
#         transform = py_vision.Normalize(mean=[-2.118, -2.036, -1.804],
#                                         std=[4.367, 4.464, 4.444])
#         return transform(img)
#     else:
#         out = (img + 1) / 2
#         return out.clamp(0, 1)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

