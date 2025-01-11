from flask import Flask, request, jsonify, render_template, send_from_directory
from chat_bot import DiagnosisBot, getSeverityDict, getDescription, getprecautionDict
import pandas as pd
import os

app = Flask(__name__)

# Define the base directory and data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(BASE_DIR, 'Data', 'Training.csv')
SYMPTOM_SEVERITY_PATH = os.path.join(BASE_DIR, 'MasterData', 'symptom_severity.csv')
SYMPTOM_DESCRIPTION_PATH = os.path.join(BASE_DIR, 'MasterData', 'symptom_Description.csv')
SYMPTOM_PRECAUTION_PATH = os.path.join(BASE_DIR, 'MasterData', 'symptom_precaution.csv')

# Load the training data
data = pd.read_csv(TRAINING_DATA_PATH)
X = data.iloc[:, :-1]  # Features (symptoms)
y = data.iloc[:, -1]   # Target (disease)

# Initialize the bot with data
bot = DiagnosisBot(data=X, target=y)

# Load additional data
getSeverityDict()
getDescription()
getprecautionDict()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({
                'status': 'error',
                'message': 'No message provided'
            }), 400

        symptoms = [s.strip() for s in user_message.split(',')]
        bot.current_symptoms = {key: 0 for key in bot.symptoms_dict.values()}  # Reset symptoms
        
        for symptom in symptoms:
            if symptom.lower() in bot.symptoms_dict:
                bot.current_symptoms[bot.symptoms_dict[symptom.lower()]] = 1
        
        response = bot.predict_disease()
        return jsonify({
            'status': 'success',
            'response': response
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/Components/NBar/<path:filename>')
def serve_nbar(filename):
    return send_from_directory(os.path.join(BASE_DIR, '../FrontEnd/Components/NBar'), filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(os.path.join(BASE_DIR, '../FrontEnd/assets'), filename)

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    print(f"Template folder path: {os.path.abspath(app.template_folder)}")
    print(f"Index.html exists: {os.path.exists(os.path.join(app.template_folder, 'index.html'))}")
    app.run(debug=True, port=5001)