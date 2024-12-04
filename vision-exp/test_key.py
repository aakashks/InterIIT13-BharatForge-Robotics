import openai

def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


OPENAI_API_KEY="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"

if check_openai_api_key(OPENAI_API_KEY):
    print("Valid OpenAI API key.")
else:
    print("Invalid OpenAI API key.")