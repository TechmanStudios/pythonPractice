from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);
