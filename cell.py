# @author AmythistHe
# @version 1.0
# @description
# @create 2021/4/17 10:43

import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.ops.composite as C
import mindspore.ops as ops
from mindspore import nn
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from utils import norm, gram_matrix
import cv2
import os

# GRADIENT_CLIP_TYPE = 1
# GRADIENT_CLIP_VALUE = 1.0
#
# clip_grad = C.MultitypeFuncGraph("clip_grad")
#
#
# @clip_grad.register("Number", "Number", "Tensor")
#
#
# def _clip_grad(clip_type, clip_value, grad):
#     """
#     Clip gradients.
#
#     Inputs:
#         clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
#         clip_value (float): Specifies how much to clip.
#         grad (tuple[Tensor]): Gradients.
#
#     Outputs:
#         tuple[Tensor], clipped gradients.
#     """
#     if clip_type not in (0, 1):
#         return grad
#     dt = F.dtype(grad)
#     if clip_type == 0:
#         new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
#                                    F.cast(F.tuple_to_array((clip_value,)), dt))
#     else:
#         new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
#     return new_grad


# class ShareCell(nn.cell):
#     def __init__(self, D, model, auto_prefix=True):
#         super(DisWithLossCell, self).__init__(auto_prefix=auto_prefix)
#         self.D = D
#         self.model = model
#
#     def construct(self, input):
#         # Train discriminator with fake data
#         recon_image = self.model(input)
#         D_fake_decision = self.D(recon_image)
#         return recon_image, D_fake_decision

class ModelWithLossCell(nn.Cell):
    def __init__(self, model, L1_loss, auto_prefix=True):
        super(ModelWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.model = model
        self.L1_loss = L1_loss

    def construct(self, input, target, bicubic):
        prediction = self.model(input)
        # if opt.residual:
        prediction = prediction + bicubic
        loss = self.L1_loss(prediction, target)
        return loss


class DisWithLossCell(nn.Cell):
    def __init__(self, D, model, BCE_loss, auto_prefix=True):
        super(DisWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.D = D
        self.model = model
        self.BCE_loss = BCE_loss
        self.print = ops.Print()

    def construct(self, input, target, real_label, fake_label):
        # Train discriminator with real data
        D_real_decision = self.D(target)
        D_real_loss = self.BCE_loss(D_real_decision, real_label)
        # Train discriminator with fake data
        recon_image = self.model(input)
        D_fake_decision = self.D(recon_image)
        # print(recon_image)
        # print(target)
        D_fake_loss = self.BCE_loss(D_fake_decision, fake_label)
        # D_real_loss 0
        # D_fake_loss  2.76310215e+01
        # self.print("D_real_decision")
        # self.print(D_real_decision)
        # self.print("D_fake_decision")
        # self.print(D_fake_decision)
        D_loss = D_real_loss + D_fake_loss
        return D_loss


class GenWithLossCell(nn.Cell):
    def __init__(self, D, model, feature_extractor, MSE_loss, BCE_loss, auto_prefix=True):
        super(GenWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.D = D
        self.model = model
        self.feature_extractor = feature_extractor
        self.MSE_loss = MSE_loss
        self.BCE_loss = BCE_loss

    def construct(self, input, target, target_ori, real_label, fake_label):
        # Train generator
        recon_image = self.model(input)
        D_fake_decision = self.D(recon_image)

        # Adversarial loss
        GAN_loss = 1e-3 * self.BCE_loss(D_fake_decision, real_label)

        # Content losses
        # print(recon_image)
        # print(target)
        mse_loss = 1e-2 * self.MSE_loss(recon_image, target)

        # Perceptual loss
        x_VGG = target_ori
        recon_VGG = F.stop_gradient(recon_image)
        real_feature = self.feature_extractor(x_VGG)
        fake_feature = self.feature_extractor(recon_VGG)
        # vgg_loss = opt.w2 * sum([self.MSE_loss(fake_feature[i], real_feature[i]) for i in range(len(real_feature))])
        # style_loss = opt.w4 * sum(
        #     [self.MSE_loss(gram_matrix(fake_feature[i]), gram_matrix(real_feature[i])) for i in
        #      range(len(real_feature))])
        vgg_loss = 0
        style_loss = 0

        for i in range(len(real_feature)):
            real = F.stop_gradient(real_feature[i])
            gram_real = F.stop_gradient(gram_matrix(real_feature[i]))
            vgg_loss += self.MSE_loss(fake_feature[i], real)
            style_loss += self.MSE_loss(gram_matrix(fake_feature[i]), gram_real)
        vgg_loss = 1e-1 * vgg_loss
        style_loss = 10 * style_loss
        # vgg_loss = 1e-1 * sum([self.MSE_loss(fake_feature[i], F.stop_gradient(real_feature[i])) for i in range(len(real_feature))])
        # style_loss = 10 * sum(
        #     [self.MSE_loss(gram_matrix(fake_feature[i]), F.stop_gradient(gram_matrix(real_feature[i]))) for i in
        #      range(len(real_feature))])

        # Back propagation
        # print("mse_loss {}".format(mse_loss))
        # print("vgg_loss {}".format(vgg_loss))
        # print("GAN_loss {}".format(GAN_loss))
        # print("style_loss {}".format(style_loss))
        G_loss = mse_loss + vgg_loss + GAN_loss + style_loss
        return G_loss

# 单步训练
class TrainOneStepCell(nn.Cell):
    def __init__(self,
                 netG,
                 netD,
                 optimizerG: nn.Optimizer,
                 optimizerD: nn.Optimizer,
                 sens=1.0,  # 1.0
                 auto_prefix=True):

        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netG.set_grad()
        self.netG.add_flags(defer_inline=True)

        self.netD = netD
        self.netD.set_grad()
        self.netD.add_flags(defer_inline=True)

        self.weights_G = optimizerG.parameters
        self.optimizerG = optimizerG
        self.weights_D = optimizerD.parameters
        self.optimizerD = optimizerD

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        # self.hyper_map = C.HyperMap()

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer_G = F.identity
        self.grad_reducer_D = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL,
                                  ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer_G = DistributedGradReducer(
                self.weights_G, mean, degree)
            self.grad_reducer_D = DistributedGradReducer(
                self.weights_D, mean, degree)

    def trainD(self, input, target, real_label, fake_label, loss, loss_net, grad,
               optimizer, weights, grad_reducer):
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = grad(loss_net, weights)(input, target, real_label, fake_label, sens)
        # grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = grad_reducer(grads)
        return F.depend(loss, optimizer(grads))

    def trainG(self, input, target, target_ori, real_label, fake_label, loss, loss_net, grad,
               optimizer, weights, grad_reducer):
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = grad(loss_net, weights)(input, target, target_ori, real_label, fake_label, sens)
        # grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = grad_reducer(grads)
        return F.depend(loss, optimizer(grads))

    def construct(self, input, target, target_ori, real_label, fake_label):
        loss_D = self.netD(input, target, real_label, fake_label)
        loss_G  = self.netG(input, target, target_ori, real_label, fake_label)
        # for p in self.netD.trainable_params():  # reset requires_grad
        #     p.requires_grad = True
        d_out = self.trainD(input, target, real_label, fake_label, loss_D, self.netD,
                            self.grad, self.optimizerD, self.weights_D,
                            self.grad_reducer_D).view(-1)
        # for p in self.netD.trainable_params():  # reset requires_grad
        #     p.requires_grad = False
        g_out = self.trainG(input, target, target_ori, real_label, fake_label, loss_G, self.netG,
                            self.grad, self.optimizerG, self.weights_G,
                            self.grad_reducer_G).view(-1)

        return d_out.mean(), g_out.mean()