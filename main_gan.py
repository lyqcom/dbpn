# @author AmythistHe
# @version 1.0
# @description
# @create 2021/4/8 18:56

from __future__ import print_function
import argparse
import os
from mindspore.train.summary import SummaryRecord
import copy
from vgg19.src.vgg import vgg19
# from output.vgg19 import Model
# from new_pirm2018 import Model
# from Discriminator_dynamic import D
from vgg19.src.config import imagenet_cfg as cfg
import mindspore
from mindspore.common import dtype as mstype
from mindspore import context, Tensor, nn
from mindspore.nn.dynamic_lr import piecewise_constant_lr
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, _InternalCallbackParam, RunContext
from mindspore.ops import operations as ops
from mindspore import context
import mindspore.dataset as ds
from dbpn_v1 import Net as DBPNLL
from dbpn_net import DBPN_NET
from dbpn import Net as DBPN
import numpy as np
from discriminator import Discriminator, FeatureExtractor, FeatureExtractorResnet
from cell import DisWithLossCell, GenWithLossCell, TrainOneStepCell
from data import get_training_set
import socket
from utils import norm, gram_matrix

# Training settings
parser = argparse.ArgumentParser(description='Mindspore Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--pretrained_iter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=3, help='Snapshots')  # 25
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')  # 1e-4
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
# parser.add_argument('--data_dir', type=str, default='/home/hoo/ms_dataset/DIV2K')
parser.add_argument('--data_dir', type=str, default='/data/DBPN_data')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
# parser.add_argument('--hr_train_dataset', type=str, default='DIV2KDemo')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--patch_size', type=int, default=60, help='Size of cropped HR image')  # 60
parser.add_argument('--pretrained_sr', default='Generator_localhost.localdomainDBPNLLPIRM_VGGVGG_46-0_1.ckpt', help='sr pretrained base model')
parser.add_argument('--load_pretrained', type=bool, default=False)
parser.add_argument('--pretrained_D', default='Discriminator_localhost.localdomainDBPNLLPIRM_VGGVGG_20-0_1.ckpt', help='sr pretrained base model')
parser.add_argument('--load_pretrained_D', type=bool, default=False)
parser.add_argument('--feature_extractor', default='VGG', help='Location to save checkpoint models')
parser.add_argument('--w1', type=float, default=1e-2, help='MSE weight')
parser.add_argument('--w2', type=float, default=1e-1, help='Perceptual weight')
parser.add_argument('--w3', type=float, default=1e-3, help='Adversarial weight')
parser.add_argument('--w4', type=float, default=10, help='Style weight')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='PIRM_VGG', help='Location to save checkpoint models')
parser.add_argument('--pretrained_path', default='./vgg19/vgg19_ImageNet.ckpt', help='Location to save checkpoint models')
# parser.add_argument('--pretrained_path', default='/home/hoo/ms_dataset/VGG/vgg16.ckpt', help='Location to save checkpoint models')

opt = parser.parse_args()
# gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
print(opt)

# 设置VGG参数
args = parser.parse_args()
args.num_classes = cfg.num_classes
args.batch_norm = cfg.batch_norm
args.has_dropout = cfg.has_dropout
args.has_bias = cfg.has_bias
args.initialize_mode = cfg.initialize_mode
args.padding = cfg.padding
args.pad_mode = cfg.pad_mode
args.weight_decay = cfg.weight_decay
args.loss_scale = cfg.loss_scale

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=5)
# context.set_context(device_target="Ascend", save_graphs=False, device_id=1)
# context.set_context(reserve_class_name_in_scope=False)

if __name__ == '__main__':
    # 创建数据集
    print('===> Loading datasets')
    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size,
                                 opt.data_augmentation)
    training_data_loader = ds.GeneratorDataset(source=train_set, column_names=["input", "target", "bicubic"],
                                               num_parallel_workers=opt.threads, shuffle=True)
    training_length = training_data_loader.get_dataset_size()
    # 标准化
    # training_data_loader = training_data_loader.map(operations=norm(),
    #                                                 input_columns=["input"])
    # training_data_loader = training_data_loader.map(operations=norm(),
    #                                                 input_columns=["target"])

    # for data in training_data_loader.create_dict_iterator():
    #     print(data["input"].shape)
    #     print(data["target"].shape)

    training_data_loader = training_data_loader.batch(opt.batchSize, drop_remainder=True)
    # print(sum(1 for _ in training_data_loader.create_dict_iterator()))

    print('===> Building model ', opt.model_type)
    if opt.model_type == 'DBPNLL':
        model = DBPNLL(num_channels=3, base_filter=64, feat=256, num_stages=10, scale_factor=opt.upscale_factor)
        # model = Model()
    # elif opt.model_type == 'DBPN-RES-MR64-3':
    # model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor)
    else:
        model = DBPN(num_channels=3, base_filter=64, feat=256, num_stages=7, scale_factor=opt.upscale_factor)

    ###Discriminator
    # D = D()
    D = Discriminator(num_channels=3, base_filter=64, image_size=opt.patch_size * opt.upscale_factor)
    # 采用VGG作为特征抽取器
    # net = vgg19(args.num_classes, args)
    # param_dict = load_checkpoint(opt.pretrained_path)
    # params_not_load = load_param_into_net(net, param_dict)
    # feature_extractor = FeatureExtractor(net)

    net = vgg19(args.num_classes, args)
    param_dict = load_checkpoint(opt.pretrained_path)
    params_not_load = load_param_into_net(net, param_dict)
    feature_extractor = FeatureExtractor(net)

    # 模型测试
    # test_Input = Tensor(np.random.rand(1, 3, 60, 60), dtype=mindspore.float32)
    test_Input = Tensor(np.full((1, 3, 60, 60), 1.5), dtype=mindspore.float32)
    # test_Target = Tensor(np.random.rand(1, 3, 240, 240), dtype=mindspore.float32)
    test_Target = Tensor(np.full((1, 3, 240, 240), 1.5), dtype=mindspore.float32)

    # 判别器测试
    # D_real_decision = D(test_Target)
    # print(D_real_decision.shape)
    # recon_image = model(test_Input)
    # D_fake_decision = D(recon_image)
    # print(D_fake_decision.shape)

    # 特征抽取器测试
    # x_VGG = test_Target
    # recon_VGG = recon_image
    # real_feature = feature_extractor(x_VGG)
    # fake_feature = feature_extractor(recon_VGG)
    # for i in real_feature:
    #     print(i.shape)
    # print("------------------------")
    # for i in fake_feature:
    #     print(i.shape)

    # 构建生成器和判别器的loss
    # 判别器：D + model + BCE_loss
    # 生成器：D + model + feature_extractor + BCE_loss + MSE_loss

    MSE_loss = nn.MSELoss()
    BCE_loss = nn.BCELoss()

    netD_with_loss = DisWithLossCell(D, model, BCE_loss)
    netG_with_loss = GenWithLossCell(D, model, feature_extractor, MSE_loss, BCE_loss)

    # 学习率衰减
    # learning rate is decayed by a factor of 10 every half of total epochs
    milestone = [int(opt.nEpochs/2), int(opt.nEpochs + 1)]
    learning_rates = [opt.lr, opt.lr/10.0]
    lr = piecewise_constant_lr(milestone, learning_rates)
    # print(lr)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)
    D_optimizer = nn.Adam(D.trainable_params(), learning_rate=lr)

    # net_train = TrainOneStepCell(netG_with_loss, netD_with_loss, optimizer,
    #                               D_optimizer)
    # manager1 = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 20, scale_factor=2, scale_window=1000)
    # manager2 = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 20, scale_factor=2, scale_window=1000)
    # scaling_sens = Tensor(np.full((1), np.finfo(np.float32).max), dtype=mstype.float32)
    myTrainOneStepCellForD = nn.TrainOneStepCell(netD_with_loss, D_optimizer)
    myTrainOneStepCellForG = nn.TrainOneStepCell(netG_with_loss, optimizer)
    # myTrainOneStepCellForD = nn.TrainOneStepWithLossScaleCell(netD_with_loss, D_optimizer, scale_sense=manager1)
    # myTrainOneStepCellForG = nn.TrainOneStepWithLossScaleCell(netG_with_loss, optimizer, scale_sense=manager2)
    #
    net_train = DBPN_NET(myTrainOneStepCellForD, myTrainOneStepCellForG)

    D.set_train()
    model.set_train()
    # feature_extractor.set_train()
    # net_train.set_train()
    # net_train.netG.feature_extractor.set_train(False)

    # Save model config
    ckpt_config = CheckpointConfig(save_checkpoint_steps=opt.snapshots)
    ckpt_cb_g = ModelCheckpoint(config=ckpt_config, directory=opt.save_folder,
                                prefix='Generator_'+hostname+opt.model_type+opt.prefix+opt.feature_extractor)
    # ckpt_cb_d = ModelCheckpoint(config=ckpt_config, directory=opt.save_folder,
    #                             prefix='Discriminator_'+hostname+opt.model_type+opt.prefix+opt.feature_extractor)

    cb_params_g = _InternalCallbackParam()
    cb_params_g.train_network = model
    cb_params_g.cur_step_num = 0
    cb_params_g.batch_num = opt.batchSize
    cb_params_g.cur_epoch_num = 0

    # cb_params_d = _InternalCallbackParam()
    # cb_params_d.train_network = D
    # cb_params_d.cur_step_num = 0
    # cb_params_d.batch_num = opt.batchSize
    # cb_params_d.cur_epoch_num = 0

    run_context_g = RunContext(cb_params_g)
    # run_context_d = RunContext(cb_params_d)
    ckpt_cb_g.begin(run_context_g)
    # ckpt_cb_d.begin(run_context_d)

    if opt.load_pretrained:
        model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
        if os.path.exists(model_name):
            param_dict = load_checkpoint(model_name)
            params_not_load = load_param_into_net(model, param_dict)
            print('Pre-trained SR model is loaded.')

    if opt.load_pretrained_D:
        D_name = os.path.join(opt.save_folder + opt.pretrained_D)
        if os.path.exists(D_name):
            param_dict = load_checkpoint(D_name)
            params_not_load = load_param_into_net(D, param_dict)
            print('Pre-trained Discriminator model is loaded.')

    # with SummaryRecord('./summary_dir', network=net) as summary_record:
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        G_epoch_loss = 0
        D_epoch_loss = 0
        feat_epoch_loss = 0
        style_epoch_loss = 0
        adv_epoch_loss = 0
        mse_epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader.create_dict_iterator(), 1):
            input = Tensor(batch["input"], dtype=mstype.float32)
            target_ori = Tensor(batch["target"], dtype=mstype.float32)
            minibatch = input.shape[0]
            ones = ops.Ones()
            zeros = ops.Zeros()
            real_label = ones((minibatch, 1), mstype.float32)  # torch.rand(minibatch,1)*0.5 + 0.7
            fake_label = zeros((minibatch, 1), mstype.float32)  # torch.rand(minibatch,1)*0.3
            input = input.asnumpy()
            target = copy.deepcopy(target_ori.asnumpy())
            for j in range(minibatch):
                input[j] = norm(input[j], vgg=True)
                target[j] = norm(target[j], vgg=True)
            input = Tensor(input, dtype=mstype.float32)
            target = Tensor(target, dtype=mstype.float32)
            # d_out, g_out, condD, scaling_sensD, condG, scaling_sensG = net_train(input, target, target_ori, real_label, fake_label)
            d_out, g_out = net_train(input, target, target_ori, real_label, fake_label)
            # log
            G_epoch_loss += d_out
            D_epoch_loss += g_out
            # feat_epoch_loss += vgg_loss
            # style_epoch_loss += style_loss
            # adv_epoch_loss += GAN_loss
            # mse_epoch_loss += mse_loss
            # if epoch==3 and iteration == 400:
            #     print('===> Saving model')
            #     cb_params_d.cur_step_num = epoch + 1
            #     cb_params_g.cur_step_num = epoch + 1
            #     ckpt_cb_g.step_end(run_context_g)
            #     ckpt_cb_d.step_end(run_context_d)
            # print("Epoch: [%2d] [%4d/%4d] G_loss: %.8f, D_loss: %.8f, condG: %d, condD: %d, scaling_sensG: %d, scaling_sensD: %d"
            #       % ((epoch), (iteration), training_length, g_out.asnumpy(), d_out.asnumpy(), condG.asnumpy(), condD.asnumpy(), scaling_sensG.asnumpy(),
            #          scaling_sensD.asnumpy()))
            print(
                "Epoch: [%2d] [%4d/%4d] G_loss: %.8f, D_loss: %.8f"
                % ((epoch), (iteration), training_length, g_out.asnumpy(), d_out.asnumpy()))
            # summary_record.add_value('scalar', 'G_loss', gout)
            # summary_record.add_value('scalar', 'D_loss', dout)
            # summary_record.record((epoch - 1) * 800 + iteration)

        print(
            "===> Epoch: [%5d] Complete: Avg. Loss G: %.4f D: %.4f" %(
                epoch, np.true_divide(G_epoch_loss.asnumpy(), training_length), np.true_divide(D_epoch_loss.asnumpy(), training_length)))
        if (epoch+1) % (opt.snapshots) == 0:
            print('===> Saving model')
            # cb_params_d.cur_step_num = epoch + 1
            cb_params_g.cur_step_num = epoch + 1
            ckpt_cb_g.step_end(run_context_g)
            # ckpt_cb_d.step_end(run_context_d)