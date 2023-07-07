from Transformer_S import *
from Transformer_T import *
from adapt import *


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BodyFaceEmotionClassifier(nn.Module):
    def __init__(self, args, frame):
        super(BodyFaceEmotionClassifier, self).__init__()
        self.args = args
        self.tr_keypoint = 70+25
        self.frame = frame

        total_features_size = 0

        # 面部每帧2048长的特征向量
        if args.use_cnn_features:
            """ add features from cnn for face """
            self.face_channel = 2048

        # 身体骨骼点：42+42+50
        if args.add_body_dnn:
            """ use simple dnn for modeling the skeleton """
            # 134
            n = 42+42+50  # 这是开放骨骼关节的数量:手21个2D点，身体25个2D点
            self.static = nn.Sequential(
                nn.Linear(n, args.first_layer_size),
                nn.ReLU()
            )
            self.body_channel = args.first_layer_size

        # 全身关键点： 137
        if args.all_features:
            """use spatial Transformer for modeling the all sklearn """
            # 137
            self.transformer = SprTransformer(num_joints=self.tr_keypoint, in_chans=2, embed_dim_ratio=256, depth=4,
                                              num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                              drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None)

            total_features_size += 256

        self.model_attention = CrossAttention(dim=256, in_channel=2048, num_heads=4, kv_bias=False,
                                         q_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)

        self.model_attention_b = CrossAttention(dim=256, in_channel=256, num_heads=4, kv_bias=False,
                                         q_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)

        self.model_tim = TempTransformer(num_frame=self.frame, num_joints=self.tr_keypoint,
                                    embed_dim_ratio=256, depth=4, num_heads=4, mlp_ratio=2.,
                                    qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                                    drop_path_rate=0.2, norm_layer=None)
        self.model_adapt = Adapt()
        self.bn1 = nn.BatchNorm1d(256)

        self.classifier_1 = nn.Sequential(nn.Linear(256, 6))
        self.classifier_2 = nn.Sequential(nn.Linear(12, 6))
        self.classifier_3 = nn.Sequential(nn.Linear(256, 6))

        self.weighted_mean = torch.nn.Conv1d(in_channels=self.tr_keypoint, out_channels=1, kernel_size=1)

        # if args.split_branches:
        #     # 在split_branches模式和分数融合中，每个类计算情感分数，然后全身和三个完全融合。
        #     """ in split_branches mode and score fusion, emotion scores are calculated per class,
        #     and then the whole body and complete fusion of three (if deep is selected) """
        #
        #     self.bn2 = nn.BatchNorm1d(total_features_size)
        #
        #     self.bn_body = nn.BatchNorm1d(args.first_layer_size)
        #     self.bn_face = nn.BatchNorm1d(2048)
        #     self.classifier_body = nn.Sequential(
        #         nn.Linear(args.first_layer_size, args.num_classes)
        #     )
        #     self.classifier_face = nn.Sequential(
        #         nn.Linear(2048, args.num_classes)
        #     )
        #
        #     """ 面部和身体的标签包含中性，整个身体不要->num_classes - 1 """
        #     if args.add_whole_body_branch:
        #         self.classifier = nn.Sequential(
        #             nn.Linear(args.first_layer_size+2048, args.num_classes-1)
        #         )
        #     else:
        #         # 如果没有全身分支与特征融合，那么就做一个简单的与分数融合
        #         self.classifier = nn.Sequential(
        #             nn.Linear(2*args.num_classes, args.num_classes - 1)
        #         )
        #     if args.do_fusion:
        #
        #         if self.args.add_whole_body_branch:
        #             self.classifier_deep = nn.Sequential(
        #                 nn.Linear(2 * args.num_classes + args.num_classes-1, args.num_classes - 1)
        #             )
        #         else:
        #             # HMT-3a
        #             self.classifier_deep = nn.Sequential(
        #                 nn.Linear(2 * args.num_classes, args.num_classes - 1)
        #             )
        #
        # else:
        #     # 只是合并所有的特征在一起，并分类与整个身体的标签
        #     # just merge all features together and classify with labels of the full body
        #     self.bn2 = nn.BatchNorm1d(total_features_size)
        #     self.bn_body = nn.BatchNorm1d(128)
        #     self.bn_face = nn.BatchNorm1d(2048)
        #
        #     self.classifier = nn.Sequential(
        #         nn.Linear(total_features_size, args.num_classes),
        #     )
        #     self.b = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18]).long()

        # if args.adapt_fusion:
        #     self.conv1 = nn.Conv2d()

    def spr_forward(self, inp, batch, get_features=False):
        # print("S_Transformer")
        face, body, hand_right, hand_left, length, facial_cnn_features = inp
        feats = []
        # 新增分支获取特征向量transformer_features_out
        if self.args.all_features:
            features = torch.cat((face, body), dim=2)     # Tensor(12, 323, 285)
            features = features.view(features.size(0), features.size(1), -1, 3)     # Tensor(12, 323, 95, 3)

            features_positions_x = features[:, :, :, 0].clone()    # Tensor(12, 323, 95)
            features_positions_y = features[:, :, :, 1].clone()

            confidences = features[:, :, :, 2].clone()

            # todo 关节置信度阈值，设为 0.1
            t = torch.Tensor([self.args.confidence_threshold]).cuda()  # threshold for confidence of joints
            # 通过阈值的值变为1，否则变为0
            confidences = (confidences > t).float() * 1

            # make all joints with threshold lower than
            # features_positions :Tensor(12, 323, 95, 2)  transformer_feature :Tensor(12, 323, 190)
            features_positions = torch.stack(
                (features_positions_x * confidences, features_positions_y * confidences), dim=3)
            # transformer_features = features_positions.view(features_positions.size(0), features_positions.size(1), -1)
            self.transformer_features_out = self.transformer.forward(features_positions)

        # body 做dnn分支，获取特征向量 static_features，args.first_layer_size
        if self.args.add_body_dnn:
            features = torch.cat((body, hand_right, hand_left), dim=2)
            features = features.view(features.size(0), features.size(1), -1, 3)

            # _x,_y,confidences分别为features第0、第1和第2列张量
            features_positions_x = features[:, :, :, 0].clone()
            features_positions_y = features[:, :, :, 1].clone()

            confidences = features[:, :, :, 2].clone()
            # 关节置信度阈值，设为 0.1
            t = torch.Tensor([self.args.confidence_threshold]).cuda()  # threshold for confidence of joints
            # 通过阈值的值变为1，否则变为0
            confidences = (confidences > t).float() * 1

            # make all joints with threshold lower than
            # 使所有关节的阀值低于
            features_positions = torch.stack(
                (features_positions_x*confidences, features_positions_y*confidences), dim=3)
            # todo 这里为什么256了？
            static_features = features_positions.view(features_positions.size(0), features_positions.size(1), -1)

            self.static_features = self.static(static_features)

        # 面部使用cnn特征分支，获得特征向量facial_cnn_features_out
        if self.args.use_cnn_features:
            self.facial_cnn_feature_out = facial_cnn_features

        cross_att_up = self.model_attention.forward(self.facial_cnn_feature_out, self.transformer_features_out, batch)
        cross_att_down = self.model_attention_b.forward(self.static_features, self.transformer_features_out, batch)

        return cross_att_up, cross_att_down

    def tim_forward(self, cross_att_1, cross_att_2):
        # print("T_Transformer")

        face_all_feature = self.model_tim.forward(cross_att_1)
        body_all_feature = self.model_tim.forward(cross_att_2)

        return face_all_feature, body_all_feature

    def forward(self, x, batch):
        cross_att_up, cross_att_down = self.spr_forward(x, batch)
        out1, out2 = self.tim_forward(cross_att_up, cross_att_down)
        # # 自适应权重
        adapt_out = self.model_adapt.forward(out1, out2)

        out_up = self.classifier_1(self.bn1(out1))
        out_down = self.classifier_1(self.bn1(out2))
        out = self.classifier_1(adapt_out)

        return out_up, out_down, out


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 in_channel,
                 num_heads=8,
                 kv_bias=False,
                 q_bias=False,
                 qk_scale=None, attn_drop=0.,
                 proj_drop=0.,
                 drop=0.5):
        # # super()继承父类的构造函数
        super().__init__()
        self.num_heads = num_heads
        self.in_channel = in_channel    #  2048

        # 对每一个head的dim
        head_dim = dim // num_heads
        # todo NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim * 1, bias=q_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 再定义一个全连接层，对应多头自注意输出b Concate拼接后，乘的W0 。输入输出节点个数都等于dim
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.conv = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1, stride=1)
        self.drop = nn.Dropout(p=drop)


    def forward(self, x, y, batch):
        x = x.unsqueeze(3)
        x = rearrange(x, 'f b c p -> (f b) c  p')
        if x.shape[1] == 2048:
            x = self.conv(x)
        y1 = rearrange(x, '(f b) c p -> f b p  c', b=323)
        x = rearrange(x, '(f b) c p -> (f b) p  c', b=323)

        Bx, Nx, Cx = x.shape     # (1, 323, 2048)
        By, Ny, Cy = y.shape     # (323, 32, 95)

        q = self.q(x).reshape(Bx, Nx, 1, self.num_heads, Cx // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(By, Ny, 2, self.num_heads, Cy // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)
        kv = (q @ k.transpose(-2, -1)) * self.scale
        kv = kv.softmax(dim=-1)
        # 对得到的结果，也就是V的权重进行dropout
        kv = self.attn_drop(kv)

        kv = (kv @ v).transpose(1, 2).reshape(By, Nx, Cy)
        # 拼接起来后还需要通过W对其进行映射，所以这里通过proj这个全连接层得到x
        kv = self.proj(kv)
        # 在进行一次dropout得到最终输出
        kv = self.proj_drop(kv)
        kv = rearrange(kv, '(b f) p c -> b f p c',  f=323)
        y1 = y1 + kv

        return y1


# if __name__ == '__main__':
#     x = torch.rand(323, 2, 2048)
#     y = torch.rand(646, 95, 256)
#     a = CrossAttention(256, 2048)
#     a.forward(x, y, 2)

