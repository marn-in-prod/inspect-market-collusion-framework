from openai import OpenAI
import os

# It automatically picks up OPENAI_API_KEY from environment variables
client = OpenAI(api_key=os.environ.get("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"))

    # Create a chat completion request
completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a short, interesting fact."},
        ]
    )

    # Print the model's response
print(completion.choices[0].message.content)