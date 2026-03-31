import anthropic
from dotenv import load_dotenv

MAX_TOKENS = 200000 
WARNING_THRESHOLD = 0.75
load_dotenv()

system_prompt = [
    "You are a senior software engineer with 20 years of experience. \
    You are direct, no nonsense, and explain things with concrete code examples. \
    You never use bullet points. You speak in plain sentences.",

    "You are a customer support agent for a coffee shop called \
    Brew & Co. You only answer questions about coffee, our menu, and \
    our store. If asked anything unrelated, politely redirect the \
    conversation back to coffee.",

    "You are a data extraction assistant. When given any text, \
    you respond ONLY with a JSON object extracting the key entities. \
    No explanation, no preamble, just raw JSON."
    ]

client = anthropic.Anthropic()

conversation_history = []

def chat_streaming(user_msg):
    conversation_history.append(
        {"role" : "user", "content" : user_msg}
    )

    full_response = ""
    input_tokens_used = 0

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system_prompt[1],
        temperature=0.5,
        messages=conversation_history
    ) as stream:

        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text 
        input_tokens_used = stream.get_final_message().usage.input_tokens
    print()

    # Token limit
    fill_percentage = input_tokens_used / MAX_TOKENS

    if fill_percentage > WARNING_THRESHOLD:
        print(f"\nWARNING: Context window {fill_percentage:.1%} full")
    else:
        print(f"\nContext window {fill_percentage:.1%} full")



    conversation_history.append(
        {"role" : "assistant", "content" : full_response}
    )

    return full_response    

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        break

    chat_streaming(user_input)