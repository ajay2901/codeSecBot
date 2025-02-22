import pytest
from src.chatbot import ChatBot

@pytest.fixture
def bot():
    return ChatBot('data\Application_Security_500_QA.csv')

def test_data_loading(bot):
    assert len(bot.df) > 0

def test_known_question(bot):
    response = bot.get_response("How do I track my order?")
    assert "order tracking portal" in response

def test_unknown_question(bot):
    response = bot.get_response("What is the meaning of life?")
    assert "I'm not sure" in response