from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import moviepy.editor as mp
import nltk
from textblob import TextBlob
import speech_recognition as sr
from langdetect import detect
import cv2
import os
from fer import FER
from scipy.io import wavfile
from twilio.rest import Client

nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['file']
    video.save(video.filename)

    filename = video.filename

    # Extract audio from video
    vid = mp.VideoFileClip(filename)
    audio = vid.audio
    audio_file_name_without_ext = filename.split('.mp4')[0]
    audio_file_name = "{}.wav".format(audio_file_name_without_ext)
    audio.write_audiofile(audio_file_name)

    # Convert audio to text
    recognizer = sr.Recognizer()
    audioFile = sr.AudioFile(audio_file_name)
    with audioFile as source:
        data = recognizer.record(source)
    text = recognizer.recognize_google(data, key=None)
    print(text)

    # Perform sentiment analysis
    from nltk.sentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    print("Sentiment score is: ", sentiment_scores)
    sentiment = ""
    if sentiment_scores["compound"] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores["compound"] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print("Sentiment: ", sentiment)

    # Perform language detection and English fluency analysis
    language = detect(text)
    if language == "en":
        blob = TextBlob(text)
        fluency = blob.correct().string
        print("Fluency: ", fluency)
    else:
        print("Language is not English.")

    # Extract frames from the video
    def extract_frames(video_name):
        # Read the video from specified path
        cam = cv2.VideoCapture(video_name)

        try:
            # Creating a folder named 'data'
            if not os.path.exists('data'):
                os.makedirs('data')
        except OSError:
            print('Error: Creating directory of data')

        # Frame rate
        frame_rate = 2

        # Calculate interval between frames
        interval = int(cam.get(cv2.CAP_PROP_FPS) / frame_rate)

        # Frame
        currentframe = 0

        while True:
            # Reading frame
            ret, frame = cam.read()

            if ret:
                if currentframe % interval == 0:
                    # If video is still left, continue creating images
                    name = './data/frame' + str(int(currentframe / 14)) + '.jpg'
                    print('Creating...' + name)

                    # Writing the extracted images
                    cv2.imwrite(name, frame)

                # Increasing counter
                currentframe += 1
            else:
                break

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

        return int(currentframe / 14 - 1)

    number_of_frames = extract_frames(filename)

    # Detect emotions in frames
def detect_emotion(frame_path):
    try:
        input_image = cv2.imread(frame_path)
        emotion_detector = FER()
        emotions = emotion_detector.detect_emotions(input_image)[0]["emotions"]
        print(emotions)
        return emotions
    except:
        print('Error.. Analyzing next frame')
        return {}
    
    emotions = {
        "angry": 0,
        "disgust": 0,
        "fear": 0,
        "happy": 0,
        "sad": 0,
        "surprise": 0,
        "neutral": 0
    }

    for i in range(number_of_frames):
        frame_path = "./data/frame" + str(i) + ".jpg"
        frame_emotions = detect_emotion(frame_path)
        for emotion, value in frame_emotions.items():
            emotions[emotion] += value

    print("Emotions:")
    print(emotions)
    max_emotion = max(emotions, key=emotions.get)
    print("Max Emotion: ", max_emotion)

    # Perform audio analysis
    frequency_sampling, audio_signal = wavfile.read(audio_file_name)

    print('\nSignal shape:', audio_signal.shape)
    print('Signal Datatype:', audio_signal.dtype)
    print('Signal duration:', round(audio_signal.shape[0] / float(frequency_sampling), 2), 'seconds')

    audio_signal = audio_signal / np.power(2, 15)

    length_signal = len(audio_signal)
    half_length = np.ceil((length_signal + 1) / 2.0).astype(np.int)

    signal_frequency = np.fft.fft(audio_signal)

    signal_frequency = abs(signal_frequency[0:half_length]) / length_signal
    signal_frequency **= 2

    len_fts = len(signal_frequency)

    if length_signal % 2:
        signal_frequency[1:len_fts] *= 2
    else:
        signal_frequency[1:len_fts - 1] *= 2

    signal_power = 10 * np.log10(signal_frequency)

    print(signal_power)
    print(signal_power.shape)

    # Prepare final data
    final_data = {
        "audio": sentiment_scores,
        "video": emotions,
        "text": text,
        "max_emotion": max_emotion
    }

    # Twilio integration
    # account_sid = 'ACa1813c9d3eb6caa63492cebdb88100fa'
    # auth_token = '4dbc15e40cacf4518cc576a6340971ac'
    # client = Client(account_sid, auth_token)
    # message_body = "Hello from Analyzer! We have analyzed your video and found that you are " + max_emotion + "."
    # message = client.messages \
    #     .create(
    #         body=message_body,
    #         from_='+15077283153',  # Replace with your Twilio phone number
    #         to='+919301658552'  # Replace with your recipient's phone number
    #     )
    print(final_data)
    return jsonify(final_data)

if __name__ == '__main__':
    app.run(debug=True)
