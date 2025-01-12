from flask import Flask, request, jsonify
import cohere

app = Flask(__name__)


co = cohere.Client(api_key="jG1adZN7sF5IYEYh1GxQcj55aQkvO20aOx0PcH1e")  

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Create chat stream
    stream = co.chat_stream(
        model='command-r-08-2024',
        message=user_message,
        temperature=0.3,
        chat_history=[
            {"role": "User", "message": "I have a really high temperature and a sore throat"},
            {"role": "Chatbot", "message": "Hello there! I am Baymax, your personal healthcare companion. You are experiencing a high temperature and a sore throat, which can indeed be uncomfortable. Let's work together to identify the possible causes and find some solutions.\n\nBased on your symptoms, there could be a few potential illnesses that you might be facing. Here are some possibilities:\n\n1. **Common Cold**: A sore throat and a high temperature are classic symptoms of the common cold. It is usually accompanied by a runny or stuffy nose, cough, and mild body aches. The cold typically resolves on its own within a week or two.\n\n2. **Strep Throat**: If your sore throat is severe and comes with a high fever, it might be a sign of strep throat. This bacterial infection often causes red and swollen tonsils, white patches at the back of the throat, and swollen lymph nodes in the neck. It requires medical attention and treatment with antibiotics.\n\n3. **Influenza (Flu)**: Flu can also cause a high fever and a sore throat, along with symptoms like body aches, fatigue, headache, and sometimes nausea and vomiting. It is highly contagious and can lead to serious complications, especially in vulnerable individuals.\n\n4. **Tonsillitis**: This condition involves inflammation of the tonsils, often causing a severe sore throat, difficulty swallowing, and sometimes bad breath. It can be caused by both viral and bacterial infections.\n\nTo determine the exact cause of your symptoms and receive appropriate treatment, I recommend that you consult a medical professional. They can examine you, perform tests if necessary, and provide you with the best course of action.\n\nIn the meantime, here are some general tips to help you feel more comfortable:\n\n- Stay hydrated by drinking plenty of fluids, especially water and warm liquids like herbal tea.\n- Get plenty of rest and try to sleep in a well-ventilated room.\n- Gargle with warm salt water to soothe your sore throat.\n- Use over-the-counter pain relievers and fever reducers, but be sure to follow the instructions and consult a pharmacist if needed.\n- Avoid smoking and exposure to secondhand smoke, as it can irritate your throat further.\n\nRemember, it is always better to seek medical advice when dealing with persistent or severe symptoms. I am here to support you and provide guidance, but a healthcare provider can offer a more accurate diagnosis and treatment plan.\n\nWould you like me to assist you in finding a nearby healthcare facility or provide any additional information to help you feel better?"}
        ],
        prompt_truncation='AUTO',
        connectors=[{"id": "web-search"}]
    )

    response_text = ""
    for event in stream:
        if event.event_type == "text-generation":
            response_text += event.text

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)