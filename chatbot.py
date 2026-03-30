import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

conversation_history = []

def chat(user_msg):
    conversation_history.append(
        {"role": "user", "content" : user_msg} 
    )

    response = client.messages.create(
        model = "claude-sonnet-4-20250514",
        max_tokens = 1000,
        system = "You are a helpful assistant", 
        messages = conversation_history
    )

    assistant_message = response.content[0].text

    conversation_history.append(
        {"role" : "assistant", "content" : assistant_message})
    
    return assistant_message, response.usage

# Chatbot
while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        break

    response_text, usage = chat(user_input)

    print(f"\nClaude: {response_text}")
    print(f"[Tokens used - Input: {usage.input_tokens} Output: {usage.output_tokens}]")
