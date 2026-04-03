import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

conversation_history = []

def chat_streaming(user_msg):
    conversation_history.append(
        {"role" : "user", "content" : user_msg}
    )

    full_response = ""

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system="You are a helpful assistant",
        temperature=1.0,
        messages=conversation_history
    ) as stream:

        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text 
    print()

    conversation_history.append(
        {"role" : "assistant", "content" : full_response}
    )

    return full_response    

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        break

    chat_streaming(user_input)