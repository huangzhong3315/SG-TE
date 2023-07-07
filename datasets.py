from utils import *
import torch.utils.data as data
import time

babyrobot_mapper = {
    "Happiness": 0,
    "Sadness": 1,
    "Surprise": 2,
    "Fear": 3,
    "Disgust": 4,
    "Anger": 5,
}


""" 获取每个数据集的分割以进行交叉验证 """
def get_db_splits(db):
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21],
                  [22, 23, 24], [25, 26, 27], [28, 29, 30]]


"""
获取所有subjects
"""
def get_all_db_subjects(db):
    """ get  subjects for each db """
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30]


""" 
    加载婴儿机器人数据集的注释 
    返回：data=[获取的v{}]，subject_to_number = {subject名：后缀数字}
"""
# 获取数据集数据时调用
def get_babyrobot_annotations():
    unique_subjects = []
    subject_to_number = {}
    data = []

    subj_idx = 0
    with open("BRED_dataset/annotations.csv") as data_file:
        # 按行读取数据，从第二行开始
        # 读完一个subject内容后读取下一个subject时，subj_idx+1
        for x in data_file.readlines()[1:]:
            # 获取数据字典
            v = {
                "path": x.split(",")[0].split(".")[0],
                "subject": x.split(",")[0].split("/")[2],
                # 将情绪与数字联系
                "emotion": babyrobot_mapper[x.split(",")[1].strip()],  # map emotion to number
                "ann_1_does_emotion_body": x.split(",")[2].strip(),
                "ann_1_does_emotion_face": x.split(",")[3].strip(),
                "ann_2_does_emotion_body": x.split(",")[4].strip(),
                "ann_2_does_emotion_face": x.split(",")[5].strip(),
                "ann_3_does_emotion_body": x.split(",")[6].strip(),
                "ann_3_does_emotion_face": x.split(",")[7].strip(),
            }

            # take as ground truth the majority
            # 以多数作为真实标签
            l = [v['ann_1_does_emotion_face'], v['ann_2_does_emotion_face'], v['ann_3_does_emotion_face']]
            # max()使用单个可迭代参数，返回其最大的项,.count返回出现次数
            # 值为yes/no，对应 yes和 no
            v['does_emotion_face'] = max(set(l), key=l.count)
            l = [v['ann_1_does_emotion_body'], v['ann_2_does_emotion_body'], v['ann_3_does_emotion_body']]
            v['does_emotion_body'] = max(set(l), key=l.count)
            # 将获取的字典追加到data[]中
            data.append(v)
            # subject为情绪状态，emotion为情感对应数字
            subject = v['subject']

            # 这里获取的是subject_number
            if subject not in unique_subjects:
                # unique_subjects = [subject_0, subject_1, ...]
                unique_subjects.append(subject)
                # subject_to_number = {'subjext_0':0, ....}
                subject_to_number[subject] = subj_idx
                subj_idx += 1

    return data, subject_to_number


# 获得训练数据和测试数据时调用
class BodyFaceDataset(data.Dataset):
    def __init__(self, args, data=None, indices=None, subjects=None, phase=None):
        self.args = args
        self.phase = phase

        # 按输入的要获取序列获取对应列表
        if args.db == "babyrobot":
            face, bodies, hands_right, hands_left, lengths, Y, Y_face, Y_body, paths, groups = data

            self.face = [face[x] for x in indices]
            self.bodies = [bodies[x] for x in indices]
            self.hands_right = [hands_right[x] for x in indices]
            self.hands_left = [hands_left[x] for x in indices]
            self.lengths = [lengths[x] for x in indices]
            self.Y = [Y[x] for x in indices]
            self.Y_face = [Y_face[x] for x in indices]
            self.Y_body = [Y_body[x] for x in indices]
            self.paths = [paths[x] for x in indices]
            self.groups = [groups[x] for x in indices]

        # 在lengths列表中追加bodies的大小（帧数）
        self.lengths = []
        for index in range(len(self.bodies)):
            self.lengths.append(self.bodies[index].shape[0])

        """
        features_path: cnn_feature
        """
        self.features = []
        for index in range(len(self.bodies)):
            features_path = self.paths[index] + "/cnn_features"
            # 加载 cnn_feature（Tensor）到cpu
            features = torch.load(features_path, map_location=lambda storage, loc: storage)
            self.features.append(features)

    def set_scaler(self, scaler):
        self.scaler = scaler

        self.hands_right = [scaler['hands_right'].transform(x) for x in self.hands_right]
        self.hands_left = [scaler['hands_left'].transform(x) for x in self.hands_left]
        self.bodies = [scaler['bodies'].transform(x) for x in self.bodies]
        self.face = [scaler['face'].transform(x) for x in self.face]

    def to_tensors(self):
        self.hands_right = [torch.from_numpy(x).float() for x in self.hands_right]
        self.hands_left = [torch.from_numpy(x).float() for x in self.hands_left]
        self.bodies = [torch.from_numpy(x).float() for x in self.bodies]
        self.face = [torch.from_numpy(x).float() for x in self.face]

    """
    将序列提前到每个数据库的最大长度序列
    """
    def prepad(self):
        """ prepad sequences to the max length sequence of each database """
        max_len = 323    # 最长帧为323
        # pad_sequence用0填充变化张量列表
        self.bodies = pad_sequence(self.bodies, batch_first=True, max_len=max_len)
        self.hands_right = pad_sequence(self.hands_right, batch_first=True, max_len=max_len)
        self.hands_left = pad_sequence(self.hands_left, batch_first=True, max_len=max_len)
        # self.faces = pad_sequence(self.faces, batch_first=True, max_len=max_len)
        self.face = pad_sequence(self.face, batch_first=True, max_len=max_len)
        self.features = pad_sequence(self.features, batch_first=True, max_len=max_len)

    def __len__(self):
        return len(self.Y)

    # 用于按键取值
    def __getitem__(self, index):
        v = time.time()

        body = self.bodies[index]
        hand_right = self.hands_right[index]
        hand_left = self.hands_left[index]
        # faces = self.faces[index]
        face = self.face[index]
        length = self.lengths[index]

        if self.args.use_cnn_features:
            features = self.features[index]
        else:
            # todo ???
            features = torch.Tensor(1)

        if self.args.db == "babyrobot":
            label_face = self.Y_face[index]
            label_body = self.Y_body[index]

        return {
            # "faces": faces,
            "face": face,
            "body": body,
            "hand_left": hand_left,
            "hand_right": hand_right,
            "label": self.Y[index],
            "label_face": label_face,
            "label_body": label_body,
            "length": length,
            "paths": self.paths[index],
            "facial_cnn_features": features
        }
