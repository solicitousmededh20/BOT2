from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os
import speech_recognition as sr
import time
from io import BytesIO
import base64
from sentence_transformers import SentenceTransformer, util
app = Flask(__name__)
CORS(app)  # Enable CORS
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class QuizBot:
    df = pd.read_excel('questionappdata\IT TECHNOLOGY (14).xlsx')

    @staticmethod
    def preprocess_data(df):
        if df.isnull().sum().sum() > 0:
            df = df.dropna()
        df.columns = df.iloc[0]
        df = df[1:]
        df['Sr. No'] = pd.to_numeric(df['Sr. No'], errors='coerce')
        df = df.dropna(subset=['Sr. No']).sort_values(by='Sr. No').reset_index(drop=True)
        df['Sr. No'] = df['Sr. No'].astype(int)
        return df
    
    @staticmethod
    def select_question(df, sub_category, level_type):
        filtered_df = df[(df['Sub - Catogory'] == sub_category) & (df['Level Type'] == level_type)]
        if filtered_df.empty:
            return None, None, None, None, None
        else:
            question_row = filtered_df.sample(n=1)
            question = question_row['Question'].values[0]
            correct_answer = question_row['Answer'].values[0]
            level_type_display = question_row['Level Type'].values[0]
            sr_no = question_row['Sr. No'].values[0]
            sub_category_display = question_row['Sub - Catogory'].values[0]
            return question, correct_answer, level_type_display, sr_no, sub_category_display
    @staticmethod
    def speech_to_text():
        recognizer = sr.Recognizer()
        print("Waiting for 15 seconds before starting speech recognition...")
        time.sleep(15)
        
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            try:
                audio = recognizer.listen(source, timeout=10)  # Stop listening after 10 seconds of speech
            except sr.WaitTimeoutError:
                print("No speech detected after 10 seconds. Stopping...")
                return ""
        
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print("Sorry, there was an error processing your request:", str(e))
            return ""
    @staticmethod
    def compute_similarity(user_answer, dataset_answer):
        user_embedding = model.encode(user_answer, convert_to_tensor=True)
        dataset_embedding = model.encode(dataset_answer, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(user_embedding, dataset_embedding)
        return similarity_score.item()

    @staticmethod
    def text_to_speech(text):
        if text:
            tts = gTTS(text=text,slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_data = audio_buffer.getvalue()
            return audio_data
        else:
            # Handle the case where no text is provided
            print("No text provided for speech synthesis.")
            return None

df = QuizBot.preprocess_data(QuizBot.df)

@app.route('/question', methods=['POST'])
def get_question():
    if request.method=='POST':
        data = request.json
        sub_category = data['sub_category']
        level_type = data['level_type']
        question, correct_answer, level_type_display, sr_no, sub_category_display = QuizBot.select_question(df, sub_category, level_type)
        
        if question is None:
            return jsonify({"error": "No questions found for the selected sub-category and level type."}), 404
        
        audio=QuizBot.text_to_speech(question)
        if audio:
            audio_base64 = base64.b64encode(audio).decode('utf-8')
        else:
            audio_base64 = ''
        return jsonify({
            "question": question,
            "correct_answer": correct_answer,
            "audio_base64":audio_base64
            
        })
    elif request.method=='GET':
        return jsonify({"massege": 'successfull'})
        

@app.route('/answer', methods=['POST'])
def check_answer():
    data = request.json
    user_answer = data['user_answer']
    correct_answer = data['correct_answer']
    similarity_score = QuizBot.compute_similarity(user_answer, correct_answer)
    similarity_score = round(similarity_score * 10, 1)

    if similarity_score < 2:
        similarity_score = 0
    
    return jsonify({
        "user_answer": user_answer,
        "correct_answer": correct_answer,
        "similarity_score": similarity_score
    })

if __name__ == "__main__":
    app.run()
