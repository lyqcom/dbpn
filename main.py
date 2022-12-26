# @author AmythistHe
# @version 1.0
# @description
# @create 2021/3/22 21:17
import argparse
import mindspore
import os
from mindspore import context, Tensor
import numpy as np
from dbpn import Net as DBPN
from dbpn_v1 import Net as DBPNLL
from dbpns import Net as DBPNS
from dbpn_iterative import Net as DBPNITER
from data import get_training_set
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore.nn.dynamic_lr import piecewise_constant_lr
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, _InternalCallbackParam, RunContext
from mindspore.ops import operations as ops
from mindspore import context, nn
from cell import ModelWithLossCell
import mindspore.dataset as ds
import pdb
import socket
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/home/hoo/ms_dataset/DIV2K')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='MIX2K_LR_aug_x4dl10DBPNITERtpami_epoch_399.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='tpami_residual_filter8', help='Location to save checkpoint models')

opt = parser.parse_args()
# gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
print(opt)

# 调用集合通信库
"""
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
init()
"""

context.set_context(device_target="Ascend", save_graphs=False, device_id=3)


if __name__ == '__main__':

    # 创建数据集
    print('===> Loading datasets')
    # 分布式
    """
    rank_id = get_rank()  # 获取当前设备在集群中的ID
    rank_size = get_group_size()  # 获取集群数量
    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
    training_data_loader = ds.GeneratorDataset(source=train_set, column_names=["input", "target", "bicubic"],
                                               num_parallel_workers=opt.threads, shuffle=True,
                                               num_shards=rank_size, shard_id=rank_id)
    """
    # 单卡
    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
    training_data_loader = ds.GeneratorDataset(source=train_set, column_names=["input", "target", "bicubic"],
                                               num_parallel_workers=opt.threads, shuffle=True)
    training_length = training_data_loader.get_dataset_size()
    training_data_loader = training_data_loader.batch(opt.batchSize, drop_remainder=True)

    # 模型载入
    print('===> Building model ', opt.model_type)
    if opt.model_type == 'DBPNLL':
        model = DBPNLL(num_channels=3, base_filter=64, feat=256, num_stages=10, scale_factor=opt.upscale_factor)
    elif opt.model_type == 'DBPN-RES-MR64-3':
        model = DBPNITER(num_channels=3, base_filter=64, feat=256, num_stages=3, scale_factor=opt.upscale_factor)
    else:
        model = DBPN(num_channels=3, base_filter=64, feat=256, num_stages=7, scale_factor=opt.upscale_factor)

    # 模型测试
    # test_Input = Tensor(np.random.rand(1, 3, 40, 40), dtype=mindspore.float32)
    # test_Target = Tensor(np.random.rand(1, 3, 320, 320), dtype=mindspore.float32)
    # test_Bicubic = Tensor(np.random.rand(1, 3, 320, 320), dtype=mindspore.float32)

    # DBPN测试
    # prediction = model(test_Input)
    # prediction = prediction + test_Bicubic
    # print(prediction.shape)

    # 模型训练
    L1_loss = nn.L1Loss()
    model_with_loss = ModelWithLossCell(model, L1_loss)

    # 学习率衰减
    # learning rate is decayed by a factor of 10 every half of total epochs
    milestone = [int(opt.nEpochs/2), int(opt.nEpochs + 1)]
    learning_rates = [opt.lr, opt.lr/10.0]
    lr = piecewise_constant_lr(milestone, learning_rates)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr, beta1=0.9, beta2=0.999, eps=1e-8)

    # 配置单步训练
    myTrainOneStepCell = nn.TrainOneStepCell(model_with_loss, optimizer)
    myTrainOneStepCell.set_train()

    # Save model config
    ckpt_config = CheckpointConfig(save_checkpoint_steps=opt.snapshots)
    ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=opt.save_folder,
                              prefix='Model_'+opt.train_dataset+hostname+opt.model_type+opt.prefix)

    cb_params = _InternalCallbackParam()
    cb_params.train_network = model
    cb_params.cur_step_num = 0
    cb_params.batch_num = opt.batchSize
    cb_params.cur_epoch_num = 0

    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)

    if opt.pretrained:
        model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
        if os.path.exists(model_name):
            param_dict = load_checkpoint(model_name)
            params_not_load = load_param_into_net(model, param_dict)
            print('Pre-trained SR model is loaded.')


    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        epoch_loss = 0

        for iteration, batch in enumerate(training_data_loader.create_dict_iterator(), 1):
            input = Tensor(batch["input"], dtype=mstype.float32)
            target = Tensor(batch["target"], dtype=mstype.float32)
            bicubic = Tensor(batch["bicubic"], dtype=mstype.float32)
            out = myTrainOneStepCell(input, target, bicubic)
            # log
            epoch_loss += out
            print("Epoch: [%2d] [%4d/%4d] loss: %.4f"
                  % ((epoch), (iteration), training_length, out.asnumpy()))
        print(
            "===> Epoch: [%5d] Complete: Avg. Loss: %.4f"
            % (epoch, np.true_divide(epoch_loss.asnumpy(), training_length)))
        if (epoch+1) % (opt.snapshots) == 0:
            print('===> Saving model')
            cb_params.cur_step_num = epoch + 1
            ckpt_cb.step_end(run_context)


