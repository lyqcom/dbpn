# @author AmythistHe
# @version 1.0
# @description
# @create 2021/4/14 10:02

from __future__ import print_function
import argparse

import os
from dbpn_v1 import Net as DBPNLL
from dbpn import Net as DBPN
import mindspore
from new_pirm2018 import Model
from mindspore.ops import composite as C
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.ops import operations as ops
from mindspore import context, Tensor, nn
import mindspore.dataset as ds
from data import get_eval_set, get_eval_fileName
from functools import reduce
import numpy as np
import time
import cv2
import utils
import pdb

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# Training settings
parser = argparse.ArgumentParser(description='Mindspore Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='Input')
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
# parser.add_argument('--test_dataset', type=str, default='PIRM_Self-Val_set')
parser.add_argument('--test_dataset', type=str, default='Set5_LR_x4')
parser.add_argument('--original_dataset', type=str, default='original')
parser.add_argument('--model_type', type=str, default='DBPNLL')
# parser.add_argument('--model', default='models/PIRM2018_region2.pth', help='sr pretrained base model')
parser.add_argument('--model', default='weights/Generator_localhost.localdomainDBPNLLPIRM_VGGVGG_47-0_1.ckpt', help='sr pretrained base model')

opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=5)


def save_img(img, img_name):
    save_img = C.clip_by_value(img.squeeze(), 0, 1).asnumpy().transpose(1, 2, 0)

    # save img
    save_dir = os.path.join(opt.output, opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = save_dir + '/' + img_name
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def x8_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.asnumpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = Tensor(tfnp, dtype=mindspore.float32)

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        return ret

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')

    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output


def chop_forward(x, model, scale, shave=16, min_size=10000, nGPUs=opt.gpus):
    b, c, h, w = x.shape
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            op = ops.Concat()
            input_batch = op(inputlist[i:(i + nGPUs)], dim=0)
            if opt.self_ensemble:
                output_batch = x8_forward(input_batch, model)
            else:
                output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.data.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


# 创建数据集
print('===> Loading datasets')
test_set = get_eval_set(os.path.join(opt.input_dir, opt.test_dataset),
                        os.path.join(opt.input_dir, opt.original_dataset),
                        opt.upscale_factor)
testing_data_loader = ds.GeneratorDataset(source=test_set, column_names=["input", "bicubic", "original"],
                                          num_parallel_workers=opt.threads, shuffle=False)
testing_data_loader = testing_data_loader.batch(opt.testBatchSize, drop_remainder=True)

# 测试集文件名
file_name = get_eval_fileName(os.path.join(opt.input_dir,opt.test_dataset))

print('===> Building model')
# if opt.model_type == 'DBPNLL':
#     model = DBPNLL(num_channels=3, base_filter=64,  feat=256, num_stages=10, scale_factor=opt.upscale_factor)
# else:
#     model = DBPN(num_channels=3, base_filter=64,  feat=256, num_stages=7, scale_factor=opt.upscale_factor)
#
# param_dict = load_checkpoint(opt.model)
# params_not_load = load_param_into_net(model, param_dict)
model = Model()
param_dict = load_checkpoint(opt.model)
params_not_load = load_param_into_net(model, param_dict)
print('Pre-trained SR model is loaded.')

psnrScore = 0
ssimScore = 0
for index, batch in enumerate(testing_data_loader.create_dict_iterator()):
    input = Tensor(batch["input"], dtype=mindspore.float32)
    original = Tensor(batch["original"], dtype=mindspore.float32)
    name = file_name[index]
    t0 = time.time()
    if opt.chop_forward:
        prediction = chop_forward(input, model, opt.upscale_factor)
    else:
        if opt.self_ensemble:
            prediction = x8_forward(input, model)
        else:
            prediction = model(input)
    t1 = time.time()
    print("===> Processing: %s || Timer: %.4f sec." % (name, (t1 - t0)))
    prediction = utils.denorm(prediction, vgg=True)
    psnrNet = nn.PSNR()
    ssimNet = nn.SSIM()
    print("-------%s PSNR----------" % name)
    print(psnrNet(original, prediction))
    psnrScore += psnrNet(original, prediction)
    print("-------%s SSIM----------" % name)
    print(ssimNet(original, prediction))
    ssimScore += ssimNet(original, prediction)
    save_img(prediction, name)
print("PSNR AVG %.4f" % np.true_divide(psnrScore.asnumpy(), 5.0))
print("SSIM AVG %.4f" % np.true_divide(ssimScore.asnumpy(), 5.0))