import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder

"""
this is to read all the extracted video features 
with .npy format into pickle file with its responding video ID
"""


class Dataset_Generator:
    def __init__(
        self,
        video_sentiments=pd.read_csv("/project/Data/Model/video_Sentiments_final.csv"),
    ):
        self.video_sentiments = video_sentiments

    def get_features(self, features_path):
        all_files = []
        for root, dirs, files in os.walk(features_path):
            for file in files:
                if file.endswith(".npy"):
                    all_files.append(os.path.join(root, file))

        all_files_np = []
        for file in all_files:
            if file.endswith(".npy"):
                feature_example = np.load(file)
                all_files_np.append((file.strip(), feature_example))
        return all_files_np

    def get_datasets(self, features_path):
        df = pd.DataFrame(
            self.get_features(features_path), columns=["Video_Name", "Video_Features"]
        )
        df["VideoID"] = df["Video_Name"].str[-24:-13]
        df = df.drop_duplicates(subset=["VideoID"], keep=False)
        video_sentiment = df.merge(self.video_sentiments, on="VideoID", how="inner")
        columns = ["VideoID", "Video_Sentiments_Score", "Video_Features"]
        video_sentiment_file = pd.DataFrame(video_sentiment, columns=columns)
        counts_dataset = video_sentiment_file["Video_Sentiments_Score"].value_counts()
        video_sentiment_file = video_sentiment_file[
            ~video_sentiment_file["Video_Sentiments_Score"].isin(
                counts_dataset[counts_dataset < 20].index
            )
        ]
        return video_sentiment_file


if __name__ == "__main__":
    video_feature = Dataset_Generator()
    # get training dataset into pickle format
    video_sentiment_train = video_feature.get_datasets(
        "/project/Data/video_feature_extractor/features_train"
    )
    file_name = "valid_train.pkl"
    f = open(file_name, "wb")
    pickle.dump(video_sentiment_train, f)
    f.close()

    # get test dataset into pickle format
    video_sentiment_test = video_feature.get_datasets(
        "/project/Data/video_feature_extractor/features_test"
    )
    file_name = "valid_test.pkl"
    f = open(file_name, "wb")
    pickle.dump(video_sentiment_test, f)
    f.close()

    # get val dataset into pickle format
    video_sentiment_val = video_feature.get_datasets(
        "/project/Data/video_feature_extractor/features_val"
    )
    file_name = "valid_val.pkl"
    f = open(file_name, "wb")
    pickle.dump(video_sentiment_val, f)
    f.close()
