from __future__ import print_function
import cv2
import math
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import pytorch_lightning as pl
from lib.model.unet_attention import ATUNet
from lib.utils.im_utils import depth2normal_torch, depth2normal
# from lib.model.loss_builder import LossBuilderHuman
# from lib.model.loss import Loss
# from typing import Any, Optional
# from pytorch_lightning.utilities.types import STEP_OUTPUT


class BaseModule(nn.Module):
    def __init__(self, im2d_in=6, im2d_out=2, im2im_in=9, im2im_out=6):
        super(BaseModule, self).__init__()

        self.img2depth = ATUNet(in_ch=im2d_in, out_ch=im2d_out)
        self.depth2color = ATUNet(in_ch=im2im_in, out_ch=im2im_out)

        # weight initialization
        for m in self.modules():
            m = weight_init_basic(m)

    def forward(self, x, pred_color=False):
        y_d = self.img2depth(x)  # (8, 32)
        output = {'depth': y_d}

        if pred_color:
            y_df, y_db = torch.chunk(y_d, chunks=2, dim=1)
            normal_f = depth2normal_torch(y_df * 255.0)
            normal_b = depth2normal_torch(y_db * 255.0)
            # without initial depth guide.
            x2 = torch.cat([x[:, :3, :, :], normal_f, normal_b], dim=1)
            y_c = self.depth2color(x2)
            output['color'] = y_c
        return output
    

# class BaseModuleS(nn.Module):
#     # shared model (fixed) -> eval() issue should be resolved.
#     def __init__(self, im2d_in=6, out_ch1=6, out_ch2=2):
#         super(BaseModuleS, self).__init__()
#
#         # shared network for depth and color
#         self.img2depth = ATUNetS(in_ch=im2d_in, out_ch1=out_ch1, out_ch2=out_ch2)
#
#         # weight initialization
#         for m in self.modules():
#             m = weight_init_basic(m)
#
#     def forward(self, x):
#         y_d = self.img2depth(x)
#         output = {'color': y_d[:, 0:6, :, :], 'depth': y_d[:, 6:8, :, :]}
#
#         return output


# class DeepHumanNetDeploy(nn.Module):
#     def __init__(self):
#         super(DeepHumanNetDeploy, self).__init__()
#         self.model = BaseModuleS()
#
#     @torch.no_grad()
#     def forward(self, x):
#         # self.model.eval()
#         return self.model(x)


class DeepHumanNet(pl.LightningModule):
    def __init__(self, opt=None):
        super(DeepHumanNet, self).__init__()
        self.model = BaseModule()
        self.automatic_optimization = False

        self.learning_rate = 0.001  # opt.learning_rate
        self.log_every_t = 200  # opt.log_every_n_steps

        # opt comes from 'human_recon_base.yaml'
        if opt is not None: # for training
            # self.loss = LossBuilderHuman(opt['batch_size'])
            self.opt_color = opt.dataset.train['opt_color']
            self.opt_depth = opt.dataset.train['opt_depth']

    def configure_optimizers(self):
        if self.opt_color is True and self.opt_depth is False:
            optimizer = optim.Adam(self.model.depth2color.parameters(), lr=self.learning_rate)
            for param in self.model.img2depth.parameters():
                param.requires_grad = False
        elif self.opt_depth is True and self.opt_color is False:
            optimizer = optim.Adam(self.model.img2depth.parameters(), lr=self.learning_rate)
            for param in self.model.depth2color.parameters():
                param.requires_grad = False
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def training_step(self, train_batch, batch_idx):
        input, gt = train_batch
        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()
        pred = self.model(input, pred_color=self.opt_color)
        train_loss, log_dict = self.loss.forward(pred,
                                                 gt,
                                                 normal=input[:, 3:6, :, :],
                                                 use_color=self.opt_color,
                                                 use_depth=self.opt_depth)

        self.manual_backward(train_loss)
        opt.step()

        # step at the last bach of each epoch.
        if self.trainer.is_last_batch:
            sch.step()

        logs = {'train_loss': train_loss}
        if batch_idx % self.log_every_t == 0:
            log_dict['input'] = input[0, :3]
            log_dict['input_guide'] = input[0, 3:6]
            input_color_grid = self.make_summary(log_dict)
            self.logger.experiment.add_scalar("Loss/Train", train_loss, self.global_step)
            self.logger.experiment.add_image("Images/Train", input_color_grid, self.global_step)
        return {'loss': train_loss, 'log': logs}

    # def on_train_epoch_end(self):
    def validation_step(self, val_batch, batch_idx):
        input, gt = val_batch
        pred = self.model(input, pred_color=self.opt_color)
        val_loss, _ = self.loss.forward(pred, gt, use_color=self.opt_color, use_depth=self.opt_depth)
        return {'loss': val_loss}

    def test_step(self, val_batch, batch_idx):
        self.model.eval()
        if torch.is_tensor(val_batch):
            return self.model(val_batch, pred_color=self.opt_color)
        else:
            input, gt = val_batch
            pred = self.model(input, pred_color=self.opt_color)
            test_loss = self.loss.forward(pred, gt, use_color=self.opt_color, use_depth=self.opt_depth)
            return {'loss': test_loss}

    @torch.no_grad()
    def in_the_wild_step(self, image, mask, guide_depth, return_color=False):
        image[mask == 0] = 1.0
        # image = image[:, :, [2, 1, 0]]
        guide_depth[mask == 0] = 0.0
        normal = depth2normal(guide_depth*255)
        image = torch.FloatTensor(image).to(self.device)
        normal = torch.FloatTensor(normal).to(self.device)
        input = torch.concatenate((image, normal), dim=2).permute(2, 0, 1).unsqueeze(0)
        input = nn.functional.interpolate(input, size=(512, 512), mode='bilinear', align_corners=True)
        return self.model(input, pred_color=return_color)

    def save_images(self, output, save_path):
        cv2.imwrite(save_path.replace('.png', '_pred_color_f.png'), output['image_front'])
        cv2.imwrite(save_path.replace('.png', '_pred_color_b.png'), output['image_back'])
        cv2.imwrite(save_path.replace('.png', '_pred_normal_f.png'), output['normal_front'] * 255)
        cv2.imwrite(save_path.replace('.png', '_pred_normal_b.png'), output['normal_back'] * 255)

    def output_to_dict(self, output, input=None, mask=None):
        if input is not None:
            image = input.permute(1, 2, 0)[:, :, :3].detach().cpu().numpy()
            normal = input.permute(1, 2, 0)[:, :, 3:].detach().cpu().numpy()
            output['input'] = (image * 255).astype(np.uint8)
            output['input_guide'] = normal[:, :, ::-1] * 255

        df, db = torch.chunk(output['depth'].squeeze(0), dim=0, chunks=2)
        if mask is not None:
            df[mask == 0] = 0
            db[mask == 0] = 0

        output['df_np'] = df[0].detach().cpu().numpy() * 255
        output['db_np'] = db[0].detach().cpu().numpy() * 255
        output['df'] = df
        output['db'] = db
        # cv2.imwrite('input.png', (image * 255).astype(np.uint8))
        # cv2.imwrite('input_guide.png', normal[:, :, ::-1] * 255)
        # cv2.imwrite('depth_front.png', df[0].detach().cpu().numpy() * 255)
        # cv2.imwrite('depth_back.png', db[0].detach().cpu().numpy() * 255)
        if 'color' in output:
            img_f, img_b = torch.chunk(output['color'].squeeze(0), dim=0, chunks=2)
            img_front = img_f.permute(1, 2, 0).detach().cpu().numpy() * 255
            img_back = img_b.permute(1, 2, 0).detach().cpu().numpy() * 255
            img_front = img_front.astype(np.uint8)
            img_back = img_back.astype(np.uint8)
            output['image_front'] = img_front
            output['image_back'] = img_back
            # cv2.imwrite('image_front.png', img_front)
            # cv2.imwrite('image_back.png', img_back)
        else:
            output['image_front'] = input[:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
            output['image_back'] = None
        return output

    def make_summary(self, log_dict, vis='normal_color'):
        log_list = []
        log_list.append(log_dict['input'][[2, 1, 0], :, :])
        log_list.append(log_dict['input_guide'])
        if 'plane' in vis:
            if 'plane_pred' in log_dict:
                p1, p2 = torch.chunk(log_dict['plane_pred'], chunks=2, dim=1)
                log_list.append(p1[0].repeat(3, 1, 1))  # front
                log_list.append(p2[0].repeat(3, 1, 1))  # back
            if 'plane_gt' in log_dict:
                p3, p4 = torch.chunk(log_dict['plane_gt'], chunks=2, dim=1)
                log_list.append(p3[0].repeat(3, 1, 1))  # front
                log_list.append(p4[0].repeat(3, 1, 1))  # back
        if 'normal' in vis:
            if 'normal_gt' in log_dict:
                p3, p4 = torch.chunk(log_dict['normal_gt'], chunks=2, dim=1)
                log_list.append(p3[0])  # front
                log_list.append(p4[0])  # back
            if 'normal_pred' in log_dict:
                p1, p2 = torch.chunk(log_dict['normal_pred'], chunks=2, dim=1)
                log_list.append(p1[0])  # front
                log_list.append(p2[0])  # back
        if 'color' in vis:
            if 'color_pred_f' in log_dict:
                log_list.append(log_dict['color_pred_f'][0][[2, 1, 0], :, :])
            if 'color_pred_b' in log_dict:
                log_list.append(log_dict['color_pred_b'][0][[2, 1, 0], :, :])
            if 'color_tgt_f' in log_dict:
                log_list.append(log_dict['color_tgt_f'][0][[2, 1, 0], :, :])
            if 'color_tgt_b' in log_dict:
                log_list.append(log_dict['color_tgt_b'][0][[2, 1, 0], :, :])

        input_color_grid = torchvision.utils.make_grid(log_list, normalize=True, scale_each=True, nrow=6)
        return input_color_grid


def weight_init_basic(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


# for test
if __name__ == '__main__':
    # input = Variable(torch.randn(4, 3, 256, 256)).float().cuda()
    input = Variable(torch.randn(4, 2, 5, 5)).float().cuda()
    _, b = torch.Tensor.chunk(input, chunks=2, dim=1)
    print(b.shape)

    print(b)
    print(input)
