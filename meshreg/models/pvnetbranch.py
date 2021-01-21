from torch import nn
import torch
from manopth import rodrigues_layer
from .resnet import resnet18
from .ransac_voting.ransac_voting_gpu import ransac_voting_layer_v3, estimate_voting_distribution_with_mean
from .domainnorm import DomainNorm

class PVNetDecoder(nn.Module):
    def __init__(self, ver_dim, seg_dim, uncertainty_pnp=True, domain_norm=False,
                 resnet_inplanes=64, fcdim=256, s32dim=256, s16dim=128, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(PVNetDecoder, self).__init__()
        self.ver_dim=ver_dim
        self.seg_dim=seg_dim
        self.uncertainty_pnp = uncertainty_pnp
        self.domain_norm = domain_norm
        if self.domain_norm:
            norm_func = DomainNorm
        else:
            norm_func = nn.BatchNorm2d

        # Randomly initialize the 1x1 Conv scoring layer
        self.fc = nn.Sequential(
            nn.Conv2d(resnet_inplanes, fcdim, 3, 1, 1, bias=False),
            norm_func(fcdim),
            nn.ReLU(True)
        )

        # x32s->512
        self.conv32s = nn.Sequential(
            nn.Conv2d(512+fcdim, s32dim, 3, 1, 1, bias=False),
            norm_func(s32dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up32sto16s = nn.UpsamplingBilinear2d(scale_factor=2)
        # x16s->256
        self.conv16s = nn.Sequential(
            nn.Conv2d(256+s32dim, s16dim, 3, 1, 1, bias=False),
            norm_func(s16dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up16sto8s = nn.UpsamplingBilinear2d(scale_factor=2)
        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+s16dim, s8dim, 3, 1, 1, bias=False),
            norm_func(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            norm_func(s4dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)
        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            norm_func(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)
        # raw->3
        self.convraw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            norm_func(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, seg_dim+ver_dim, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, DomainNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Override initialization for the only Conv2D with normal ReLU activation
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


    def decode_keypoint(self, output):
        vertex = output['vertex'].permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex.shape
        vertex = vertex.view(b, h, w, vn_2//2, 2)
        mask = torch.argmax(output['seg'], 1)
        if self.uncertainty_pnp:
            mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
            kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)
            output.update({'mask': mask, 'kpt_2d': kpt_2d, 'var': var})
        else:
            kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
            output.update({'mask': mask, 'kpt_2d': kpt_2d})


    def forward(self, input, encoder_output, encoder_features, compute_pnp=False):
        # Get encoder features
        x2s = encoder_features["res_conv1_relu"]
        x4s = encoder_features["res_layer1"]
        x8s = encoder_features["res_layer2"]
        x16s = encoder_features["res_layer3"]
        x32s = encoder_features["res_layer4"]
        xfc = self.fc(x32s)

        fm = self.conv32s(torch.cat([xfc, x32s], 1))
        fm = self.up32sto16s(fm)

        fm = self.conv16s(torch.cat([fm, x16s], 1))
        fm = self.up16sto8s(fm)

        fm = self.conv8s(torch.cat([fm, x8s], 1))
        fm = self.up8sto4s(fm)
        if fm.shape[2]==136:
            fm = nn.functional.interpolate(fm, (135,180), mode='bilinear', align_corners=False)

        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)

        x=self.convraw(torch.cat([fm,input],1))
        seg_pred=x[:,:self.seg_dim,:,:]
        ver_pred=x[:,self.seg_dim:,:,:]

        ret = {'seg': seg_pred, 'vertex': ver_pred}

        if compute_pnp:
            with torch.no_grad():
                self.decode_keypoint(ret)
        return ret