# Ads Sentiment Prediction

## Goal: 
This project aims to predict the sentiments of advertising videos by utilizing deep learning techniques. It is considered to be a very challenging task due to the complexity of the visual world. For example, oftentimes a website might ask a user to verify one is human by selecting the correct visual prompt from a CAPTCHA; these quick checks give high confidence that a real human and not a bot is accessing the website, because it is difficult for computers to accurately interpret the patterns of image or videos. 

## Approach: 

To enable this research, the study uses a dataset of 1,992 advertising videos downloaded from YouTube and rich annotation of sentiments, funny and exciting scores. The project extracted video features using a pre-trained model-ResNet101, and modeled videos with a gated recurrent unit (GRU) and long short-term memory (LSTM) network. This study was able to achieve a 36.8% accuracy rate with GRU and 38.5% with LSTM on predicting sentiments, 81.05% accuracy on predicting funny or not, and 76.72% on classifying exciting or not exciting.

### Train vs Validation loss & Accuracy for model of predicting sentiment of video ad – LSTM
<img src="https://github.com/JolieFang/Ads_Sentiment_Predictor/blob/main/7_Results/Train_vs_Validation_LSTM.png" width="500" height="570">

### Train vs Validation loss & Accuracy for model of predicting funny or not of video ad
<img src="https://github.com/JolieFang/Ads_Sentiment_Predictor/blob/main/7_Results/Train_vs_Validation_Funny_not.png" width="510" height="410">

### Confusion Matrix on Model of Predicting Funny or Not
<img src="https://github.com/JolieFang/Ads_Sentiment_Predictor/blob/main/7_Results/CM_%20Predicting_Funny_Or_Not.png" width="510" height="410">

Lastly, created a backend server where the best-performing model was loaded and served to make real-time predictions through HTTP requests using FastAPI and developed a Streamlit web app to upload test set data and download the model predictions easily.

### Streamlit Web App
<img src="https://github.com/JolieFang/Ads_Sentiment_Predictor/blob/main/7_Results/Streamlit.png" width="540" height="440">

## Tools and Libraries used:
- Python
- Pytorch Lightning
- Ray 
- Torchvision
- FastAPI
- Streamlit
- COnsistent RAnk Logits for Ordinal regression
- LSTM
- GRU
- Transfer learning

## Dataset Source:
Hussain, Z., Zhang, M., Zhang, X., Ye, K., Thomas, C., Agha, Z., Ong, N. and Kovashka, A., 2017. Automatic understanding of image and video advertisements. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1705-1715).

## Features Extraction:
Miech, A. (2022). Fast and Easy to use video feature extractor. [online] GitHub. Available at: https://github.com/antoine77340/video_feature_extractor/blob/master/README.md [Accessed 3 Aug. 2022].

