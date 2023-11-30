
import os
from collections import OrderedDict

import timm
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from timm.models.swin_transformer import SwinTransformer
from timm.models.convnext import ConvNeXt

from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation
from model.non_local import NonLocalBlock
from tools import util

from mmseg.models.decode_heads import FPNHead
from mmseg.ops import Upsample, resize


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)

        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)
        self.self_att = False

        if args.aggregation in ["gem", "spoc", "mac", "rmac"]:
            if args.l2 == "before_pool":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":
                self.aggregation = nn.Sequential(self.aggregation, Flatten())
        
        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim
        if args.non_local:
            non_local_list = [NonLocalBlock(channel_feat=get_output_channels_dim(self.backbone),
                                           channel_inner=args.channel_bottleneck)]* args.num_non_local
            self.non_local = nn.Sequential(*non_local_list)
            self.self_att = True

    def forward(self, x):
        x = self.backbone(x)

        if self.self_att:
            x = self.non_local(x)

        x = self.aggregation(x)

        return x


class SwinBackbone(SwinTransformer):

    def __init__(self,
                 backbone_name: str,
                 depths=(2,2,6,2),
                 img_size=224,
                 window_size=7,
                 embed_dim=96,
                 num_heads=(3, 6, 12, 24),
                 patch_size=4,
                 patch_norm=True,
                 norm_layer=nn.LayerNorm):
        super().__init__(img_size=img_size, patch_size=patch_size, patch_norm=patch_norm, window_size=window_size,
                         depths=depths, norm_layer=norm_layer,embed_dim=embed_dim,num_heads=num_heads)

        self.depths = depths
        self.multi_feature = []
        self.out_layer_idx  = self.get_multi_layer_out(depths)

        # load pretrained params
        original_model = timm.create_model(backbone_name, pretrained=True)
        self.load_state_dict(original_model.state_dict(), strict=False)

        self.query_patch_embed = util.PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3,
                                            embed_dim=self.embed_dim,
                                            norm_layer=norm_layer if patch_norm else None)

        self.dataset_patch_embed = util.PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=5,
                                              embed_dim=self.embed_dim,
                                              norm_layer=norm_layer if patch_norm else None)

    @staticmethod
    def get_multi_layer_out(depths):
        out_layer_idx = list(depths)
        cache = 0
        for idx, i in enumerate(depths):
            cache += i
            out_layer_idx[idx] = cache
        return out_layer_idx

    def load_state_dict(self, state_dict, strict=True):

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith('patch_embed'):
                new_state_dict[k] = v

        super().load_state_dict(new_state_dict, strict=strict)

    def forward_features(self, x):
        if x.shape[1] == 3:
            x = self.query_patch_embed(x)
        elif x.shape[1] == 5:
            x = self.dataset_patch_embed(x)
        else:
            raise ValueError(f'input tensor channel should be 3 or 5, but get {x.shape[1]} now')

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        multi_feature = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in range(self.num_layers):
                B, L, C = x.shape
                multi_feature.append(x.view(B, int(L**0.5), int(L**0.5), C).permute(0,3,1,2))

            x = layer(x)
        self.multi_feature = multi_feature
        assert len(self.multi_feature) == len(self.depths)

        x = self.norm(x)  # B L C
        return x


    def forward(self, x):
        x = self.forward_features(x)
        return x


class FPNHeadPooling(FPNHead):
    def __init__(self, feature_dim=None, **kwargs):
        """
        fuse multi-layer features
        :param feature_dim: (int) channel_nums we want to get after going through this module
        :param kwargs: some important args needed pass in
            - in_channels (int|Sequence[int]): Input channels.
            - channels (int): Channels after modules, before unify-channel which is a middle tmp channel
            - num_classes (int): Number of classes. default=1000 [dont use]
            - feature_strides (tuple[int]): The strides for input feature maps.
                stack_lateral. All strides suppose to be power of 2. The first
                one is of largest resolution.
        """
        super().__init__(**kwargs)
        self.conv_unify = nn.Conv2d(self.channels, feature_dim, kernel_size=1)
        pass


    def unify_channel(self,x):
        out = self.conv_unify(x)
        return out


    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = self.unify_channel(output) # B,C,H,W
        output = torch.flatten(output,start_dim=2).permute(0,2,1) # BCHW -> BCL -> BLC
        return output


class ConvNeXtBackbone(ConvNeXt):
    def __init__(self,
                 backbone_name:str,
                 depths=(3,3,9,3),
                 dims=(96,192,384,768)):
        super().__init__(depths=depths, dims=dims)

        original_model = timm.create_model(backbone_name, pretrained=True)
        self.load_state_dict(original_model.state_dict(), strict=False)

    def forward(self, x):
        # Drop the head
        x = self.forward_features(x)
        return x


def get_aggregation(args):
    if args.aggregation == "gem":
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "spoc":
        return aggregation.SPoC()
    elif args.aggregation == "mac":
        return aggregation.MAC()
    elif args.aggregation == "rmac":
        return aggregation.RMAC()
    elif args.aggregation == "netvlad":
        return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim,
                                   work_with_tokens=args.work_with_tokens)
    elif args.aggregation == 'crn':
        return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=args.features_dim)
    elif args.aggregation == "rrm":
        return aggregation.RRM(args.features_dim)
    elif args.aggregation == 'none'\
            or args.aggregation == 'cls' \
            or args.aggregation == 'seqpool':
        return nn.Identity()


def get_backbone(args):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = args.backbone.startswith('swin')

    if args.backbone.startswith("swin"):
        if args.backbone.endswith("tiny"):
            # check input image size
            assert args.resize[0] == 224
            model_cfg = dict(depths=(2, 2, 6, 2),
                             window_size=7, img_size=224,
                             embed_dim=96, num_heads=(3,6,12,24))
            backbone = SwinBackbone('swin_tiny_patch4_window7_224', **model_cfg)
            args.features_dim = 96 * 8
        elif args.backbone.endswith("small"):
            assert args.resize[0] == 224
            model_cfg = dict(depths=(2, 2, 18, 2),
                             window_size=7, img_size=224,
                             embed_dim=96,num_heads=(3,6,12,24))
            backbone = SwinBackbone('swin_small_patch4_window7_224', **model_cfg)
            args.features_dim = 96 * 8
        elif args.backbone.endswith("base"):
            assert args.resize[0] == 384 or args.resize[0] == 224
            if args.resize[0] == 384:
                model_cfg = dict(depths=(2, 2, 18, 2),
                                 window_size=12, img_size=384,
                                 embed_dim=128, num_heads=(4, 8, 16, 32))
                backbone = SwinBackbone('swin_base_patch4_window12_384', **model_cfg)
            else:
                model_cfg = dict(depths=(2, 2, 18, 2),
                                 window_size=7, img_size=224,
                                 embed_dim=128, num_heads=(4, 8, 16, 32))
                backbone = SwinBackbone('swin_base_patch4_window7_224_in22k', **model_cfg)
            args.features_dim = 128 * 8
        else:
            raise NotImplementedError(f"The interface of {args.backbone} is not implemented")
        return backbone

    elif args.backbone.startswith("convnext"):
        if args.backbone.endswith("tiny"):
            # check input image size
            assert args.resize[0] == 224,  f'Input size should be either 224 or 384, but get {args.resize[0]}'
            model_cfg = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
            backbone = ConvNeXtBackbone('convnext_tiny', **model_cfg)
            args.features_dim = 96 * 8
        elif args.backbone.endswith("small"):
            assert args.resize[0] == 224,  f'Input size should be either 224 or 384, but get {args.resize[0]}'
            model_cfg = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
            backbone = ConvNeXtBackbone('convnext_small', **model_cfg)
            args.features_dim = 96 * 8
        elif args.backbone.endswith("base"):
            assert args.resize[0] == 384, f'Input size should be either 224 or 384, but get {args.resize[0]}'
            model_cfg = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
            backbone = ConvNeXtBackbone('convnext_base', **model_cfg)
            args.features_dim = 128 * 8
        else:
            raise NotImplementedError(f"The interface of {args.backbone} is not implemented")
        return backbone


def get_output_channels_dim(model, type:str= 'feat'):
    """Return the number of channels in the output of a model."""
    if type == 'feat':
        return model(torch.ones([1, 3, 224, 224])).shape[1]
    elif type == 'token':
        return model(torch.ones([1, 3, 224, 224])).shape[2]
    else:
        raise Exception(f'type Err: which should be feat or token but get {type}')

