import os

from openai import OpenAI


def load_dotenv(path=".env"):
  if not os.path.exists(path):
    return
  with open(path, "r") as handle:
    for raw_line in handle:
      line = raw_line.strip()
      if not line or line.startswith("#") or "=" not in line:
        continue
      key, value = line.split("=", 1)
      os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
if not API_KEY:
  raise ValueError("Missing OPEN_ROUTER_API_KEY in environment or .env file.")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=API_KEY,
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="qwen/qwen3-vl-30b-a3b-thinking",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)