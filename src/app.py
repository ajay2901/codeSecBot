from flask import Flask, request, jsonify
from .chatbot import ChatBot

# Initialize chatbot
bot = ChatBot('A:\\chatbot\\CodeSecBot\\data\\Application_Security_500_QA.csv')

# CLI Version
def cli_chat():
    print("ChatBot: Hi! How can I help you today? (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = bot.get_response(user_input)
        print(f"ChatBot: {response}")

# Web Version
app = Flask(__name__)

@app.route('/')
def home():
    return """
    <h1>ChatBot</h1>
    <form action="/ask" method="post">
        <input type="text" name="query" placeholder="Ask me something...">
        <button type="submit">Ask</button>
    </form>
    """

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    response = bot.get_response(query)
    return jsonify({'query': query, 'response': response})

if __name__ == '__main__':
    # Run CLI: python -m src.app
    # Run Web: FLASK_APP=src/app flask run
    cli_chat()  # Comment this line if you want to run web version