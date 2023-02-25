# club house

![image](https://user-images.githubusercontent.com/24197201/221328592-f0e0bc20-2d1c-4b41-8056-ebea76f625c5.png)

we have used deep neural network to automate event description and summarization from pictures being uploaded on our university page.

# Requirements

1.Python

2.Tensorflow,keras

3.flicker8k dataset (kaggle)

4.google-colab

5.google-drive

6.FAST-API

# pipeline

image will be fed to caption generator that will be generating caption.The generator is combinator of CNN+LSTM trained of flicker8k dataset.Idea is to crunch captions out of all pictures generated via caption generator within 30 days and fed to summarizer.summarizer is pre-trained network which will summarize the content and push a subscription based news letter to subscriber to the current website.

# result

currently frontend and backend both still under developement and are still in pre-integeration phase.Custom model for caption generator is struggling to gain accuracy enough to be proceeded and hence being re trained for better accuracy.

# code
code.py and hack.py will be the actual implementation of idea using different deep learning models.Both will generate summary for news letter.hack.py use ResNet50 and other uses incecptionV3
