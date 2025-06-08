from openai import OpenAI
import pandas as pd


client = OpenAI(
    base_url='https://qianfan.baidubce.com/v2',
    api_key='bce-v3/ALTAK-C7ttIT7iikByn5dJvkyGk/4ea4b6f49292c87df9548b5dd8cb2392ae153e54'
)
chat_completion = client.chat.completions.create(
    model="ernie-4.0-8k-latest", 
    messages=[
    {
        "role": "user",
        "content": "您好",

    }
]
)
print(chat_completion)
chat_completion.choices[0].message.content

df_test = pd.read_excel('data/test.xlsx')




from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)