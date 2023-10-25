import torch.nn as nn
import torch
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from .cross_attention import CrossAttention
from .cross_attention import CrossAttentionWithoutPE


class ResBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self,  in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True))
        self.actv = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.main(x) + x
        out = self.actv(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spec_norm=False, LR=0.01, stride=1, up=False):
        super(ConvBlock, self).__init__()
        self.up = up
        if self.up:
            self.up_smaple = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up_smaple = None
        if spec_norm:
            self.main = nn.Sequential(
                spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False)),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True),
            )

    def forward(self, x1, x2=None):
        if self.up_smaple is not None:
            x1 = self.up_smaple(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.main(x)
        else:
            return self.main(x1)


class ResBlockNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockNet, self).__init__()
        self.main = list()
        self.main.append(ResBlock(in_channels, out_channels))
        self.main.append(ResBlock(out_channels, out_channels))
        self.main.append(ResBlock(out_channels, out_channels))
        self.main.append(ResBlock(out_channels, out_channels))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x) + x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, spec_norm=False, LR=0.2):
        super(Encoder, self).__init__()
        self.layer1 = ConvBlock(in_channels, 16, spec_norm, LR=LR) # 256
        self.layer2 = ConvBlock(16, 16, spec_norm, LR=LR) # 256
        self.layer3 = ConvBlock(16, 32, spec_norm, stride=2, LR=LR) # 128
        self.layer4 = ConvBlock(32, 32, spec_norm, LR=LR) # 128
        self.layer5 = ConvBlock(32, 64, spec_norm, stride=2, LR=LR) # 64
        self.layer6 = ConvBlock(64, 64, spec_norm, LR=LR) # 64
        self.layer7 = ConvBlock(64, 128, spec_norm, stride=2, LR=LR) # 32
        self.layer8 = ConvBlock(128, 128, spec_norm, LR=LR) # 32
        self.layer9 = ConvBlock(128, 256, spec_norm, stride=2, LR=LR) # 16
        self.layer10 = ConvBlock(256, 256, spec_norm, LR=LR) # 16
        self.down_sampling = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        feature_map1 = self.layer1(x)
        feature_map2 = self.layer2(feature_map1)
        feature_map3 = self.layer3(feature_map2)
        feature_map4 = self.layer4(feature_map3)
        feature_map5 = self.layer5(feature_map4)
        feature_map6 = self.layer6(feature_map5)
        feature_map7 = self.layer7(feature_map6)
        feature_map8 = self.layer8(feature_map7)
        feature_map9 = self.layer9(feature_map8)
        feature_map10 = self.layer10(feature_map9)
        feature_list = [feature_map1,
                        feature_map2,
                        feature_map3,
                        feature_map4,
                        feature_map5,
                        feature_map6,
                        feature_map7,
                        feature_map8,
                        feature_map9,
                        # feature_map10,
                        ]
        output = feature_map10

        return output, feature_list


class RefEncoder(nn.Module):
    def __init__(self, in_channels=3, spec_norm=False, LR=0.2):
        super(RefEncoder, self).__init__()
        self.layer1 = ConvBlock(in_channels, 16, spec_norm, LR=LR)  # 256
        self.layer2 = ConvBlock(16, 16, spec_norm, LR=LR)  # 256
        self.layer3 = ConvBlock(16, 32, spec_norm, stride=2, LR=LR)  # 128
        self.layer4 = ConvBlock(32, 32, spec_norm, LR=LR)  # 128
        self.layer5 = ConvBlock(32, 64, spec_norm, stride=2, LR=LR)  # 64
        self.layer6 = ConvBlock(64, 64, spec_norm, LR=LR)  # 64
        self.layer7 = ConvBlock(64, 128, spec_norm, stride=2, LR=LR)  # 32
        self.layer8 = ConvBlock(128, 128, spec_norm, LR=LR)  # 32
        self.layer9 = ConvBlock(128, 256, spec_norm, stride=2, LR=LR)  # 16
        self.layer10 = ConvBlock(256, 256, spec_norm, LR=LR)  # 16
        self.down_sampling = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        feature_map1 = self.layer1(x)
        feature_map2 = self.layer2(feature_map1)
        feature_map3 = self.layer3(feature_map2)
        feature_map4 = self.layer4(feature_map3)
        feature_map5 = self.layer5(feature_map4)
        feature_map6 = self.layer6(feature_map5)
        feature_map7 = self.layer7(feature_map6)
        feature_map8 = self.layer8(feature_map7)
        feature_map9 = self.layer9(feature_map8)
        feature_map10 = self.layer10(feature_map9)
        feature_list = [feature_map1,
                        feature_map2,
                        feature_map3,
                        feature_map4,
                        feature_map5,
                        feature_map6,
                        feature_map7,
                        feature_map8,
                        feature_map9,
                        feature_map10,
                        ]
        output = feature_map10
        return output, feature_list


class ReconstructionDecoder(nn.Module):
    def __init__(self, spec_norm=False, LR=0.2):
        super(ReconstructionDecoder, self).__init__()
        self.layer10 = ConvBlock(256, 256, spec_norm, LR=LR)  # 16->16
        self.layer9 = ConvBlock(256 + 256, 256, spec_norm, LR=LR)  # 16->16
        self.layer8 = ConvBlock(256 + 128, 128, spec_norm, LR=LR, up=True)  # 16->32
        self.layer7 = ConvBlock(128 + 128, 128, spec_norm, LR=LR)  # 32->32
        self.layer6 = ConvBlock(128 + 64, 64, spec_norm, LR=LR, up=True)  # 32-> 64
        self.layer5 = ConvBlock(64 + 64, 64, spec_norm, LR=LR)  # 64 -> 64
        self.layer4 = ConvBlock(64 + 32, 32, spec_norm, LR=LR, up=True)  # 64 -> 128
        self.layer3 = ConvBlock(32 + 32, 32, spec_norm, LR=LR)  # 128 -> 128
        self.layer2 = ConvBlock(32 + 16, 16, spec_norm, LR=LR, up=True)  # 128 -> 256
        self.layer1 = ConvBlock(16 + 16, 16, spec_norm, LR=LR)  # 256 -> 256
        self.last_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, feature_list):
        feature_map10 = self.layer10(x)
        feature_map9 = self.layer9(torch.cat([feature_map10, feature_list[-1]], dim=1))
        feature_map8 = self.layer8(feature_map9, feature_list[-2])
        feature_map7 = self.layer7(torch.cat([feature_map8, feature_list[-3]], dim=1))
        feature_map6 = self.layer6(feature_map7, feature_list[-4])
        feature_map5 = self.layer5(torch.cat([feature_map6, feature_list[-5]], dim=1))
        feature_map4 = self.layer4(feature_map5, feature_list[-6])
        feature_map3 = self.layer3(torch.cat([feature_map4, feature_list[-7]], dim=1))
        feature_map2 = self.layer2(feature_map3, feature_list[-8])
        feature_map1 = self.layer1(torch.cat([feature_map2, feature_list[-9]], dim=1))
        feature_map0 = self.last_conv(feature_map1)

        recons_feature_list = [feature_map10,
                               feature_map9,
                               feature_map8,
                               feature_map7,
                               feature_map6,
                               feature_map5,
                               feature_map4,
                               feature_map3,
                               feature_map2,
                               feature_map1,
                               feature_map0]

        return self.tanh(feature_map0), recons_feature_list


class SelfAttentionOnlyContrastiveBi(nn.Module):
    '''
    This is a demo for how to choose id for patch-wise contrastive loss
    If you want to use it, you must incorporate it into an attention module which is applied on the sketch features or
    discard the convolutional layers.
    Otherwise the gradient of convolutional layers cannot be effectively calculated.
    '''
    def __init__(self, in_dim=256, num_selected_points=128):
        super(SelfAttentionOnlyContrastiveBi, self).__init__()
        self.channel_in = in_dim
        self.num_selected_points = num_selected_points
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1, padding=0, bias=False)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sketch_features, transformer_features):

        B, C, H, W = sketch_features.size()
        proj_query = self.query_conv(sketch_features).view(B, -1, H * W).permute(0, 2, 1)  # (B, H * W, C)
        proj_key = self.key_conv(sketch_features).view(B, -1, H * W)  # (B, C, H * W)

        energy = torch.bmm(proj_query, proj_key)  # (B, H * W, H * W)
        attention = self.softmax(energy)  # (B, H * W, H * W)
        attention_var = torch.var(attention, unbiased=True, dim=-1)  # (B, H * W)
        _, indexes = torch.sort(attention_var, descending=True, dim=-1)  # (B, H * W)
        contrastive_indexes = [indexes[:, -self.num_selected_points:], indexes[:, :self.num_selected_points]]

        return _, contrastive_indexes


class MixDecoder(nn.Module):
    def __init__(self, spec_norm=False, LR=0.2):
        super(MixDecoder, self).__init__()
        self.corr_module_1 = CrossAttentionWithoutPE(4, 4, 256, 0.1, 1024)
        self.corr_module_2 = CrossAttentionWithoutPE(4, 2, 128, 0.1, 1024)
        self.corr_module_3 = CrossAttentionWithoutPE(4, 1, 64, 0.1, 1024)
        self.self_attention_module_1 = SelfAttentionOnlyContrastiveBi(256, 128)
        self.layer10 = ConvBlock(256, 256, spec_norm, LR=LR)  # 16->16
        self.layer9 = ConvBlock(256 + 256, 256, spec_norm, LR=LR)  # 16->16
        self.layer8 = ConvBlock(256 + 128, 128, spec_norm, LR=LR, up=True)  # 16->32
        self.layer7 = ConvBlock(128 + 128, 128, spec_norm, LR=LR)  # 32->32
        self.layer6 = ConvBlock(128 + 64, 64, spec_norm, LR=LR, up=True)  # 32-> 64
        self.layer5 = ConvBlock(64 + 64, 64, spec_norm, LR=LR)  # 64 -> 64
        self.layer4 = ConvBlock(64 + 32, 32, spec_norm, LR=LR, up=True)  # 64 -> 128
        self.layer3 = ConvBlock(32 + 32, 32, spec_norm, LR=LR)  # 128 -> 128
        self.layer2 = ConvBlock(32 + 16, 16, spec_norm, LR=LR, up=True)  # 128 -> 256
        self.layer1 = ConvBlock(16 + 16, 16, spec_norm, LR=LR)  # 256 -> 256
        self.last_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, recons_feature_list, ref_feature_list):

        b, ch, h, w = recons_feature_list[0].size()  # [_, 256, 16, 16]
        sketch_features_1 = recons_feature_list[0].reshape(b, ch, h * w)
        sketch_features_1 = sketch_features_1.permute(0, 2, 1)

        b, ch, h, w = ref_feature_list[9].size()  # [_, 256, 16, 16]
        reference_features_1 = ref_feature_list[9].reshape(b, ch, h * w)
        # reference_features_1 = ref_feature_1.reshape(b, ch, h * w)
        reference_features_1 = reference_features_1.permute(0, 2, 1)

        transformer_features_1 = self.corr_module_1(sketch_features_1, reference_features_1)
        transformer_features_1 = transformer_features_1.permute(0, 2, 1)
        transformer_features_1 = transformer_features_1.reshape(b, ch, h, w)

        # id
        temp_sketch_features_1 = recons_feature_list[0]
        contrastive_patch_ids = None
        _, contrastive_patch_ids = self.self_attention_module_1(temp_sketch_features_1, transformer_features_1)

        feature_map10 = self.layer10(transformer_features_1)  # [_, 256, 16, 16]
        # recons_feature_list[1] [_, 256, 16, 16]
        feature_map9 = self.layer9(torch.cat([feature_map10, recons_feature_list[1]], dim=1))  # [_, 256, 16, 16]
        # recons_feature_list[2] [_, 128, 32, 32]
        feature_map8 = self.layer8(feature_map9, recons_feature_list[2])  # [_, 128, 32, 32]
        # temp_sketch_features_2 = feature_map8

        b, ch, h, w = feature_map8.size()  # [_, 128, 32, 32]
        feature_map8 = feature_map8.reshape(b, ch, h * w)
        feature_map8 = feature_map8.permute(0, 2, 1)

        b, ch, h, w = ref_feature_list[7].size()  # [_, 128, 32, 32]
        reference_features_2 = ref_feature_list[7].reshape(b, ch, h * w)
        # reference_features_2 = ref_feature_2.reshape(b, ch, h * w)
        reference_features_2 = reference_features_2.permute(0, 2, 1)

        transformer_features_2 = self.corr_module_2(feature_map8, reference_features_2)
        transformer_features_2 = transformer_features_2.permute(0, 2, 1)
        transformer_features_2 = transformer_features_2.reshape(b, ch, h, w)  # [_, 128, 32, 32]

        # recons_feature_list[3] [_, 128, 32, 32]
        feature_map7 = self.layer7(torch.cat([transformer_features_2, recons_feature_list[3]], dim=1))  # [_, 128, 32, 32]
        # recons_feature_list[4] [_, 64, 64, 64]
        feature_map6 = self.layer6(feature_map7, recons_feature_list[4])  # [_, 64 ,64, 64]

        # temp_sketch_features_3 = feature_map6

        b, ch, h, w = ref_feature_list[5].size()  # [_, 64, 64, 64]
        reference_features_3 = ref_feature_list[5].reshape(b, ch, h * w)
        # reference_features_3 = ref_feature_3.reshape(b, ch, h * w)
        reference_features_3 = reference_features_3.permute(0, 2, 1)

        b, ch, h, w = feature_map6.size()
        feature_map6 = feature_map6.reshape(b, ch, h * w)
        feature_map6 = feature_map6.permute(0, 2, 1)

        transformer_features_3 = self.corr_module_3(feature_map6, reference_features_3)  # [_, 64 ,64, 64]
        transformer_features_3 = transformer_features_3.permute(0, 2, 1)
        transformer_features_3 = transformer_features_3.reshape(b, ch, h, w)

        # recons_feature_list[5] # [_, 64 ,64, 64]
        feature_map5 = self.layer5(torch.cat([transformer_features_3, recons_feature_list[5]], dim=1))  # [_, 64 ,64, 64]
        # recons_feature_list[6] # [_, 32 ,128, 128]
        feature_map4 = self.layer4(feature_map5, recons_feature_list[6])  # [_, 32 ,128, 128]
        # recons_feature_list[7] [_, 32 ,128, 128]
        feature_map3 = self.layer3(torch.cat([feature_map4, recons_feature_list[7]], dim=1))  # [_, 32 ,128, 128]
        # recons_feature_list[8] # [_, 16 ,256, 256]
        feature_map2 = self.layer2(feature_map3, recons_feature_list[8])  # [_, 16 ,256, 256]
        # recons_feature_list[9] # [_, 16 ,256, 256]
        feature_map1 = self.layer1(torch.cat([feature_map2, recons_feature_list[9]], dim=1))  # [_, 16 ,256, 256]
        feature_map0 = self.last_conv(feature_map1)  # [_, 3 ,256, 256]

        return self.tanh(feature_map0), contrastive_patch_ids


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class ContrastiveMlp(nn.Module):
    def __init__(self, in_features=None, hidden_features=None, out_features=None, act_layer=nn.LeakyReLU(), num_patches=64, use_mlp=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.l2norm = Normalize(2)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.mlp_net = nn.Sequential(*[self.fc1, self.act, self.fc2])
        # self.num_patches = num_patches
        self.use_mlp = use_mlp

    def forward(self, feat, patch_id):
        # feat (B, C, H, W)
        b, ch, h, w = feat.size()
        features = feat.reshape(b, ch, h * w).permute(0, 2, 1)  # (B,HxW,C)
        # Input Shape (B,HxW,C)

        x_sample = features[torch.arange(b)[:, None], patch_id, :].flatten(0, 1)  # (BxK, C) K is the number of patch

        if self.use_mlp:
            x_sample = self.mlp_net(x_sample)

        x_sample = self.l2norm(x_sample)

        return x_sample


class ReferenceGenerator(nn.Module):
    def __init__(self, spec_norm=False, LR=0.2):
        super(ReferenceGenerator, self).__init__()
        self.reference_encoder = RefEncoder(in_channels=3, spec_norm=spec_norm, LR=LR)
        self.sketch_encoder = Encoder(in_channels=3, spec_norm=spec_norm, LR=LR)
        self.reconstruction_decoder = ReconstructionDecoder()
        self.mix_decoder = MixDecoder()
        self.res_model_1 = ResBlockNet(256, 256)
        self.mapping_encoder = ContrastiveMlp(in_features=256, hidden_features=256, out_features=256, num_patches=128)

    def forward(self, reference, sketch, gt_img):
        # Encoder
        v_r, ref_feature_list = self.reference_encoder(reference)
        v_s, sketch_feature_list = self.sketch_encoder(sketch)

        # Reconstruction Decoder
        recons_res_features = self.res_model_1(v_s)
        recons_image, recons_feature_list = self.reconstruction_decoder(recons_res_features, sketch_feature_list)

        # Transfer Color
        output_image, contrastive_patch_ids = self.mix_decoder(recons_feature_list, ref_feature_list)

        # Please refer to class SelfAttentionOnlyContrastiveBi for the right usage of RCS
        # Contrastive Learning　1
        gt_features, _ = self.reference_encoder(gt_img)
        feat_k_pool_0 = self.mapping_encoder(gt_features, contrastive_patch_ids[0])  # key
        res_output_features, _ = self.reference_encoder(output_image)
        feat_q_pool_0 = self.mapping_encoder(res_output_features, contrastive_patch_ids[0])  # query

        # Contrastive Learning　2
        feat_k_pool_1 = self.mapping_encoder(gt_features, contrastive_patch_ids[1])  # key
        feat_q_pool_1 = self.mapping_encoder(res_output_features, contrastive_patch_ids[1])  # query

        feat_q_pool = [feat_q_pool_0, feat_q_pool_1]
        feat_k_pool = [feat_k_pool_0, feat_k_pool_1]

        return output_image, recons_image, v_r, v_s, feat_q_pool, feat_k_pool

