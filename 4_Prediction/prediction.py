import torch as th
from torch.nn.utils.rnn import (
    pack_padded_sequence,
)
import math
from Load_video_for_prediction import Video_Reader
from model import get_model
from preprocessing import Preprocessing
import torch.nn.functional as F
from Ads_Model import AdsLSTM

# from sklearn.preprocessing import LabelEncoder


class VideosPredictor:
    def __init__(
        self,
        checkpoint="/Users/joliefang/Desktop/ads/experiments/exp1_17-10-2022--13-41-26/version_0/checkpoints/epoch=44-step=135.ckpt",
        batch_size=64,
        model=None,
        preprocess=None,
        video_reader=None
        # label_encoder=None
    ):
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.model = model or get_model()
        self.lstm_model = self.load_model()
        self.preprocess = preprocess or Preprocessing()
        self.video_reader = video_reader or Video_Reader()
        # self.label_encoder = label_encoder or pickle.load(open('label_encoder.pkl','rb'))

    def load_model(self):
        lstm_model = AdsLSTM.load_from_checkpoint(self.checkpoint)
        lstm_model.eval()
        return lstm_model

    def get_video_features(self, video_path):
        video_tensor = self.video_reader.getitem(video_path)
        video = self.preprocess(video_tensor)
        n_chunk = len(video)
        features = th.FloatTensor(n_chunk, 2048).fill_(0)
        n_iter = int(math.ceil(n_chunk / float(self.batch_size)))

        for i in range(n_iter):
            min_ind = i * self.batch_size
            max_ind = (i + 1) * self.batch_size
            video_batch = video[min_ind:max_ind].cpu()
            batch_features = self.model(video_batch)
            batch_features = F.normalize(batch_features, dim=1)
            features[min_ind:max_ind] = batch_features

        features = features.detach().cpu().numpy()
        features = features.astype("float16")
        features = th.from_numpy(features)
        return features

    # predict with the model
    def get_prediction(self, video_path):
        video_features = self.get_video_features(video_path)
        batched_video_features = video_features.unsqueeze(0).float()
        video_features_packed = pack_padded_sequence(
            batched_video_features,
            [batched_video_features.size(1)],
            batch_first=True,
            enforce_sorted=False,
        )

        pred = self.lstm_model(video_features_packed)
        softmax_pred = th.softmax(pred, axis=-1)
        predicted_class = th.argmax(softmax_pred).item()
        class_mapping = [
            "active",
            "alarmed",
            "alert",
            "amazed",
            "amused",
            "calm",
            "cheerful",
            "creative",
            "eager",
            "educated",
            "inspired",
            "persuaded",
        ]
        predicted = class_mapping[predicted_class]
        return predicted


if __name__ == "__main__":
    video_predictor = VideosPredictor()
    pred = video_predictor.get_prediction(
        "/Users/joliefang/Desktop/ads/test/Predication/New_Videos/Alwaysperiodpoverty2021.mp4"
    )
    print(pred)
