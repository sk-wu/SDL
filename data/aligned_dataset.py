import numpy as np
import os
import torch
import random
import imageio
from skimage import util, feature
from skimage.color import rgb2gray
from PIL import Image
from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_transform_without_tps_aug, \
    get_transform_mode_1, get_transform_mode_2, get_transform_mode_3, get_transform_mode_1_contrastive, \
    get_transform_mode_2_contrastive, get_transform_mode_3_contrastive, get_transform_new_ours


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        if opt.primitive != "seg_edges":
            dir_A = "_" + opt.primitive
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))
            # self.A = Image.open(self.A_paths[0])
            self.A_paths = self.A_paths[:opt.max_dataset_size]
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + "_seg")
            self.A_paths = sorted(make_dataset(self.dir_A))
            # the seg input will be saved as "A"
            self.A = Image.open(self.A_paths[0])
            self.dir_A_edges = os.path.join(opt.dataroot, opt.phase + "_edges")
            if not os.path.exists(self.dir_A_edges):
                os.mkdir(self.dir_A_edges)
            self.A_paths_edges = sorted(make_dataset(self.dir_A_edges))
            if not os.path.exists(self.dir_A_edges):
                os.makedirs(self.dir_A_edges)
            self.A_edges = Image.open(self.A_paths_edges[0]) if self.A_paths_edges else None

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))
            # self.B = Image.open(self.B_paths[0]).convert('RGB')
            self.B_paths = self.B_paths[:opt.max_dataset_size]
            if opt.primitive == "seg_edges" and not self.A_edges:
                self.A_edges = Image.fromarray(util.invert(feature.canny(rgb2gray(np.array(self.B)), sigma=0.5)))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))
        self.dataset_size = len(self.A_paths)

    def adjust_input_size(self, opt, A, B):
        """
        change image size once when loading the image.
        :return:
        """
        # TODO: if we use instance map, support resize method = NEAREST
        ow, oh = A.size
        # for cuda memory capacity
        if max(ow, oh) > 1000:
            ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none' or opt.resize_or_crop == "crop":
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        A = A.resize((new_w, new_h), Image.NEAREST)
        if opt.primitive == "seg_edges":
            self.A_edges = self.A_edges.resize((new_w, new_h), Image.NEAREST)
        if self.opt.isTrain:
            B = B.resize((new_w, new_h), Image.BICUBIC)
        return A, B

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index])
        B = Image.open(self.B_paths[index]).convert('RGB')
        A, B = self.adjust_input_size(self.opt, A, B)
        params = get_params(self.opt, B.size, B)
        reference_tensor = None
        prob = random.random()

        if prob < self.opt.split_prob:
            if self.opt.label_nc == 0:
                transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=False)
                A_img = A.convert('RGB')
                A_tensor = transform_A(A_img)
                if self.opt.primitive == "seg_edges":
                    # apply transforms only on the edges and then fuse it to the seg
                    transform_A_edges = get_transform(self.opt, params, is_primitive=True, is_edges=True)
                    A_edges = self.A_edges.convert('RGB')
                    A_edges_tensor = transform_A_edges(A_edges)
                    if self.opt.canny_color == 0:
                        A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                    else:
                        A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()
            else:
                transform_A = get_transform(self.opt, params, normalize=False, is_primitive=True)
                A_tensor = transform_A(A) * 255.0

            B_tensor = inst_tensor = feat_tensor = 0
            ### input B (real images)
            if self.opt.isTrain or self.opt.use_encoded_image:
                # B = self.B
                # transform_B = get_transform(self.opt, params, is_primitive=False)
                # B_tensor = transform_B(B)
                # transform_reference = get_transform_without_tps_aug(self.opt, params, is_primitive=False)
                # reference_tensor = transform_reference(B)

                transform_reference, transform_B = get_transform_mode_1(self.opt, params, is_primitive=False)
                B_tensor = transform_B(B)
                reference_tensor = transform_reference(B)

                transform_contrastive = get_transform_mode_1_contrastive(self.opt, params, is_primitive=False)
                contrastive_tensor_list = []
                for _ in range(self.opt.batchSize):
                    contrastive_tensor_list.append(transform_contrastive(B))

            ### if using instance maps
            if not self.opt.no_instance:
                inst_path = self.inst_paths[index]
                inst = Image.open(inst_path)
                inst_tensor = transform_A(inst)

                if self.opt.load_features:
                    feat_path = self.feat_paths[index]
                    feat = Image.open(feat_path).convert('RGB')
                    norm = normalize()
                    feat_tensor = norm(transform_A(feat))

            input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'image_contrastive': contrastive_tensor_list,
                          'feat': feat_tensor, 'reference': reference_tensor, 'path': self.A_paths[0]}

            return input_dict

        elif prob >= self.opt.split_prob and prob < 2 * self.opt.split_prob:
            if self.opt.label_nc == 0:
                transform_A = get_transform_without_tps_aug(self.opt, params, is_primitive=True, is_edges=False)
                A_img = A.convert('RGB')
                A_tensor = transform_A(A_img)
                if self.opt.primitive == "seg_edges":
                    # apply transforms only on the edges and then fuse it to the seg
                    transform_A_edges = get_transform(self.opt, params, is_primitive=True, is_edges=True)
                    A_edges = self.A_edges.convert('RGB')
                    A_edges_tensor = transform_A_edges(A_edges)
                    if self.opt.canny_color == 0:
                        A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                    else:
                        A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()
            else:
                transform_A = get_transform(self.opt, params, normalize=False, is_primitive=True)
                A_tensor = transform_A(A) * 255.0

            B_tensor = inst_tensor = feat_tensor = 0
            ### input B (real images)
            if self.opt.isTrain or self.opt.use_encoded_image:
                # B = self.B
                # transform_B = get_transform_without_tps_aug(self.opt, params, is_primitive=False)
                # B_tensor = transform_B(B)
                # transform_reference = get_transform(self.opt, params, is_primitive=False)
                # reference_tensor = transform_reference(B)

                transform_reference, transform_B = get_transform_mode_2(self.opt, params, is_primitive=False)
                B_tensor = transform_B(B)
                reference_tensor = transform_reference(B)

                transform_contrastive = get_transform_mode_2_contrastive(self.opt, params, is_primitive=False)
                contrastive_tensor_list = []
                for _ in range(self.opt.batchSize):
                    contrastive_tensor_list.append(transform_contrastive(B))

            ### if using instance maps
            if not self.opt.no_instance:
                inst_path = self.inst_paths[index]
                inst = Image.open(inst_path)
                inst_tensor = transform_A(inst)

                if self.opt.load_features:
                    feat_path = self.feat_paths[index]
                    feat = Image.open(feat_path).convert('RGB')
                    norm = normalize()
                    feat_tensor = norm(transform_A(feat))

            input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'image_contrastive': contrastive_tensor_list,
                          'feat': feat_tensor, 'reference': reference_tensor, 'path': self.A_paths[0]}

            return input_dict

        else:
            if self.opt.label_nc == 0:
                transform_A, tps_change_edge = get_transform_new_ours(self.opt, params, is_primitive=True, is_edges=False)
                A_img = A.convert('RGB')
                A_tensor = transform_A(A_img)
                if self.opt.primitive == "seg_edges":
                    # apply transforms only on the edges and then fuse it to the seg
                    transform_A_edges, tps_change_edge = get_transform_new_ours(self.opt, params, is_primitive=True, is_edges=True)
                    A_edges = self.A_edges.convert('RGB')
                    A_edges_tensor = transform_A_edges(A_edges)
                    if self.opt.canny_color == 0:
                        A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                    else:
                        A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()
            else:
                transform_A, tps_change_edge = get_transform_new_ours(self.opt, params, normalize=False, is_primitive=True)
                A_tensor = transform_A(A) * 255.0

            B_tensor = inst_tensor = feat_tensor = 0
            ### input B (real images)
            if self.opt.isTrain or self.opt.use_encoded_image:
                # B = self.B
                # transform_B = get_transform(self.opt, params, is_primitive=False)
                # B_tensor = transform_B(B)
                #
                # params_bianxing = get_params(self.opt, B.size, B)
                # transform_bianxing = get_transform(self.opt, params_bianxing, is_primitive=True, is_edges=False)
                # reference_tensor = transform_bianxing(B)

                params_bianxing = get_params(self.opt, B.size, B)
                transform_reference, transform_B = get_transform_mode_3(self.opt, params, params_bianxing, is_primitive=False, tps_change_edge=tps_change_edge)

                # change color
                temp_B = torch.from_numpy(np.asarray(B)).float()
                prob_data_aug = random.random()
                noise_data_aug = torch.randn(1)*128
                noise_data_aug = noise_data_aug.expand_as(temp_B[:, :, 0])

                if prob_data_aug >= 0.1 and prob_data_aug < 0.4:
                    temp_B[:, :, 0] += noise_data_aug
                elif prob_data_aug >= 0.4 and prob_data_aug < 0.7:
                    temp_B[:, :, 1] += noise_data_aug
                elif prob_data_aug >= 0.7 and prob_data_aug <= 1:
                    temp_B[:, :, 2] += noise_data_aug

                temp_B = torch.clip(temp_B, 0, 255).numpy()
                temp_B = Image.fromarray(np.uint8(temp_B))

                B_tensor = transform_B(temp_B)
                reference_tensor = transform_reference(temp_B)

                transform_contrastive = get_transform_mode_3_contrastive(self.opt, params, is_primitive=False, tps_change_edge=tps_change_edge)
                contrastive_tensor_list = []
                for _ in range(self.opt.batchSize):
                    temp_style_B = torch.from_numpy(np.asarray(B)).float()
                    prob_data_aug = random.random()
                    noise_data_aug = torch.randn(1) * 128
                    noise_data_aug = noise_data_aug.expand_as(temp_style_B[:, :, 0])
                    if prob_data_aug >= 0 and prob_data_aug < 0.34:
                        temp_style_B[:, :, 0] += noise_data_aug
                    elif prob_data_aug >= 0.34 and prob_data_aug < 0.67:
                        temp_style_B[:, :, 1] += noise_data_aug
                    elif prob_data_aug >= 0.67 and prob_data_aug <= 1:
                        temp_style_B[:, :, 2] += noise_data_aug

                    temp_style_B = torch.clip(temp_style_B, 0, 255).numpy()
                    temp_style_B = Image.fromarray(np.uint8(temp_style_B))
                    contrastive_tensor_list.append(transform_contrastive(temp_style_B))

            ### if using instance maps
            if not self.opt.no_instance:
                inst_path = self.inst_paths[index]
                inst = Image.open(inst_path)
                inst_tensor = transform_A(inst)

                if self.opt.load_features:
                    feat_path = self.feat_paths[index]
                    feat = Image.open(feat_path).convert('RGB')
                    norm = normalize()
                    feat_tensor = norm(transform_A(feat))

            input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'image_contrastive': contrastive_tensor_list,
                          'feat': feat_tensor, 'reference': reference_tensor, 'path': self.A_paths[0]}

            return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'


class AlignedDataset_1(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        if opt.primitive != "seg_edges":
            dir_A = "_" + opt.primitive
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))
            # self.A = Image.open(self.A_paths[0])
            self.A_paths = self.A_paths[:opt.max_dataset_size]
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + "_seg")
            self.A_paths = sorted(make_dataset(self.dir_A))
            # the seg input will be saved as "A"
            self.A = Image.open(self.A_paths[0])
            self.dir_A_edges = os.path.join(opt.dataroot, opt.phase + "_edges")
            if not os.path.exists(self.dir_A_edges):
                os.mkdir(self.dir_A_edges)
            self.A_paths_edges = sorted(make_dataset(self.dir_A_edges))
            if not os.path.exists(self.dir_A_edges):
                os.makedirs(self.dir_A_edges)
            self.A_edges = Image.open(self.A_paths_edges[0]) if self.A_paths_edges else None

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))
            # self.B = Image.open(self.B_paths[0]).convert('RGB')
            self.B_paths = self.B_paths[:opt.max_dataset_size]
            if opt.primitive == "seg_edges" and not self.A_edges:
                self.A_edges = Image.fromarray(util.invert(feature.canny(rgb2gray(np.array(self.B)), sigma=0.5)))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))
        self.dataset_size = len(self.A_paths)

    def adjust_input_size(self, opt, A, B):
        """
        change image size once when loading the image.
        :return:
        """
        # TODO: if we use instance map, support resize method = NEAREST
        ow, oh = A.size
        # for cuda memory capacity
        if max(ow, oh) > 1000:
            ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none' or opt.resize_or_crop == "crop":
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        A = A.resize((new_w, new_h), Image.NEAREST)
        if opt.primitive == "seg_edges":
            self.A_edges = self.A_edges.resize((new_w, new_h), Image.NEAREST)
        if self.opt.isTrain:
            B = B.resize((new_w, new_h), Image.BICUBIC)
        return A, B

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index])
        B = Image.open(self.B_paths[index]).convert('RGB')
        A, B = self.adjust_input_size(self.opt, A, B)
        params = get_params(self.opt, B.size, B)
        reference_tensor = None
        prob = random.random()

        if prob < self.opt.split_prob:
            if self.opt.label_nc == 0:
                transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=False)
                A_img = A.convert('RGB')
                A_tensor = transform_A(A_img)
                if self.opt.primitive == "seg_edges":
                    # apply transforms only on the edges and then fuse it to the seg
                    transform_A_edges = get_transform(self.opt, params, is_primitive=True, is_edges=True)
                    A_edges = self.A_edges.convert('RGB')
                    A_edges_tensor = transform_A_edges(A_edges)
                    if self.opt.canny_color == 0:
                        A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                    else:
                        A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()
            else:
                transform_A = get_transform(self.opt, params, normalize=False, is_primitive=True)
                A_tensor = transform_A(A) * 255.0

            B_tensor = inst_tensor = feat_tensor = 0
            ### input B (real images)
            if self.opt.isTrain or self.opt.use_encoded_image:
                # B = self.B
                # transform_B = get_transform(self.opt, params, is_primitive=False)
                # B_tensor = transform_B(B)
                # transform_reference = get_transform_without_tps_aug(self.opt, params, is_primitive=False)
                # reference_tensor = transform_reference(B)

                transform_reference, transform_B = get_transform_mode_1(self.opt, params, is_primitive=False)
                B_tensor = transform_B(B)
                reference_tensor = transform_reference(B)

                transform_contrastive = get_transform_mode_1_contrastive(self.opt, params, is_primitive=False)
                contrastive_tensor_list = []
                for _ in range(self.opt.batchSize):
                    contrastive_tensor_list.append(transform_contrastive(B))

            ### if using instance maps
            if not self.opt.no_instance:
                inst_path = self.inst_paths[index]
                inst = Image.open(inst_path)
                inst_tensor = transform_A(inst)

                if self.opt.load_features:
                    feat_path = self.feat_paths[index]
                    feat = Image.open(feat_path).convert('RGB')
                    norm = normalize()
                    feat_tensor = norm(transform_A(feat))

            input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'image_contrastive': contrastive_tensor_list,
                          'feat': feat_tensor, 'reference': reference_tensor, 'path': self.A_paths[0]}
            return input_dict

        elif prob >= self.opt.split_prob and prob < 2 * self.opt.split_prob:
            if self.opt.label_nc == 0:
                transform_A = get_transform_without_tps_aug(self.opt, params, is_primitive=True, is_edges=False)
                A_img = A.convert('RGB')
                A_tensor = transform_A(A_img)
                if self.opt.primitive == "seg_edges":
                    # apply transforms only on the edges and then fuse it to the seg
                    transform_A_edges = get_transform(self.opt, params, is_primitive=True, is_edges=True)
                    A_edges = self.A_edges.convert('RGB')
                    A_edges_tensor = transform_A_edges(A_edges)
                    if self.opt.canny_color == 0:
                        A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                    else:
                        A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()
            else:
                transform_A = get_transform(self.opt, params, normalize=False, is_primitive=True)
                A_tensor = transform_A(A) * 255.0

            B_tensor = inst_tensor = feat_tensor = 0
            ### input B (real images)
            if self.opt.isTrain or self.opt.use_encoded_image:
                # B = self.B
                # transform_B = get_transform_without_tps_aug(self.opt, params, is_primitive=False)
                # B_tensor = transform_B(B)
                # transform_reference = get_transform(self.opt, params, is_primitive=False)
                # reference_tensor = transform_reference(B)

                transform_reference, transform_B = get_transform_mode_2(self.opt, params, is_primitive=False)
                B_tensor = transform_B(B)
                reference_tensor = transform_reference(B)

                transform_contrastive = get_transform_mode_2_contrastive(self.opt, params, is_primitive=False)
                contrastive_tensor_list = []
                for _ in range(self.opt.batchSize):
                    contrastive_tensor_list.append(transform_contrastive(B))

            ### if using instance maps
            if not self.opt.no_instance:
                inst_path = self.inst_paths[index]
                inst = Image.open(inst_path)
                inst_tensor = transform_A(inst)

                if self.opt.load_features:
                    feat_path = self.feat_paths[index]
                    feat = Image.open(feat_path).convert('RGB')
                    norm = normalize()
                    feat_tensor = norm(transform_A(feat))

            input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'image_contrastive': contrastive_tensor_list,
                          'feat': feat_tensor, 'reference': reference_tensor, 'path': self.A_paths[0]}
            return input_dict

        else:
            if self.opt.label_nc == 0:
                # transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=self.opt.primitive == "edges")
                transform_A, tps_change_edge = get_transform_new_ours(self.opt, params, is_primitive=True, is_edges=False)
                A_img = A.convert('RGB')
                A_tensor = transform_A(A_img)
                if self.opt.primitive == "seg_edges":
                    # apply transforms only on the edges and then fuse it to the seg
                    transform_A_edges, tps_change_edge = get_transform_new_ours(self.opt, params, is_primitive=True, is_edges=True)
                    A_edges = self.A_edges.convert('RGB')
                    A_edges_tensor = transform_A_edges(A_edges)
                    if self.opt.canny_color == 0:
                        A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                    else:
                        A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()
            else:
                transform_A, tps_change_edge = get_transform_new_ours(self.opt, params, normalize=False, is_primitive=True)
                A_tensor = transform_A(A) * 255.0

            B_tensor = inst_tensor = feat_tensor = 0
            ### input B (real images)
            if self.opt.isTrain or self.opt.use_encoded_image:
                # B = self.B
                # transform_B = get_transform(self.opt, params, is_primitive=False)
                # B_tensor = transform_B(B)
                #
                # params_bianxing = get_params(self.opt, B.size, B)
                # transform_bianxing = get_transform(self.opt, params_bianxing, is_primitive=True, is_edges=False)
                # reference_tensor = transform_bianxing(B)

                params_bianxing = get_params(self.opt, B.size, B)
                transform_reference, transform_B = get_transform_mode_3(self.opt, params, params_bianxing, is_primitive=False, tps_change_edge=tps_change_edge)
                B_tensor = transform_B(B)
                reference_tensor = transform_reference(B)

                transform_contrastive = get_transform_mode_3_contrastive(self.opt, params, is_primitive=False, tps_change_edge=tps_change_edge)
                contrastive_tensor_list = []
                for _ in range(self.opt.batchSize):
                    contrastive_tensor_list.append(transform_contrastive(B))

            ### if using instance maps
            if not self.opt.no_instance:
                inst_path = self.inst_paths[index]
                inst = Image.open(inst_path)
                inst_tensor = transform_A(inst)

                if self.opt.load_features:
                    feat_path = self.feat_paths[index]
                    feat = Image.open(feat_path).convert('RGB')
                    norm = normalize()
                    feat_tensor = norm(transform_A(feat))

            input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'image_contrastive': contrastive_tensor_list,
                          'feat': feat_tensor, 'reference': reference_tensor, 'path': self.A_paths[0]}

            return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_1'


# TODO : fix test loader as well with scale adjustment
class AlignedDataset_test(BaseDataset):
    def initialize(self, opt):
        print("in initialize")
        self.opt = opt
        self.root = opt.dataroot

        if opt.vid_input:
            print(os.path.join(opt.dataroot + opt.phase))
            reader = imageio.get_reader(os.path.join(opt.dataroot + opt.phase), 'ffmpeg')
            opt.phase = 'vid_frames'
            dir_A = "_" + opt.primitive if self.opt.primitive != "seg_edges" else '_seg'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            if not os.path.exists(self.dir_A):
                os.mkdir(self.dir_A)
            i = 0
            for im in reader:
                print(i)
                if i == 240:
                    break
                imageio.imwrite("%s/%d.png" % (self.dir_A, i), im)
                i += 1

        ### input A (label maps)
        dir_A = "_" + opt.primitive if self.opt.primitive != "seg_edges" else '_seg'
        # Top-10
        # dir_A = "_A"
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (reference images)
        dir_B = '_B'
        # Top-10
        # dir_B = '_references'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        if opt.primitive == "seg_edges":
            self.dir_A_edges = os.path.join(opt.dataroot, opt.phase + "_edges")
            if not os.path.exists(self.dir_A_edges):
                os.makedirs(self.dir_A_edges)
            self.A_paths_edges = sorted(make_dataset(self.dir_A_edges))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)
        print("dataset_size", self.dataset_size)

    def adjust_input_size(self, opt, A, B):
        """
        change image size once when loading the image.
        :return:
        """
        # TODO: if we use instance map, support resize method = NEAREST
        ow, oh = A.size
        # for cuda memory capacity
        if max(ow, oh) > 1000:
            ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none' or opt.resize_or_crop == "crop":
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        A = A.resize((new_w, new_h), Image.NEAREST)
        if opt.primitive == "seg_edges":
            self.A_edges = self.A_edges.resize((new_w, new_h), Image.NEAREST)
        if self.opt.isTrain:
            B = B.resize((new_w, new_h), Image.BICUBIC)
        return A, B

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index])
        B = Image.open(self.B_paths[index]).convert('RGB')
        A, B = self.adjust_input_size(self.opt, A, B)
        params = get_params(self.opt, B.size, B)

        if self.opt.label_nc == 0:
            # transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=self.opt.primitive == "edges")
            transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=False)
            A_img = A.convert('RGB')
            A_tensor = transform_A(A_img)
            if self.opt.primitive == "seg_edges":
                # apply transforms only on the edges and then fuse it to the seg
                transform_A_edges = get_transform(self.opt, params, is_primitive=True, is_edges=True)
                A_edges = self.A_edges.convert('RGB')
                A_edges_tensor = transform_A_edges(A_edges)
                if self.opt.canny_color == 0:
                    A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                else:
                    A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()

        else:
            transform_A = get_transform(self.opt, params, normalize=False, is_primitive=True)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)

        transform_B = get_transform(self.opt, params, is_primitive=False)
        B_tensor = transform_B(B)
        reference_tensor = transform_B(B)
        # transform_reference = get_transform_without_tps_aug(self.opt, params, is_primitive=False)
        # reference_tensor = transform_reference(B)

        ### if using instance maps
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'reference': reference_tensor, 'path': self.A_paths[index]}

        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_test'
