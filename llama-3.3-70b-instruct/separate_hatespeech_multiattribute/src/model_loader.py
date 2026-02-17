import json
import urllib.request
import urllib.error

import config


class OpenRouterClient:
    def __init__(self, api_key, model, base_url, app_name=None, http_referer=None):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.http_referer = http_referer

    def chat(self, messages, max_tokens=10, temperature=0.1, stop=None, logprobs=False):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.app_name:
            headers["X-Title"] = self.app_name
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = str(exc)
            raise RuntimeError(f"OpenRouter error: {error_body}") from exc


def load_model():
    print("Connecting to OpenRouter...")
    client = OpenRouterClient(
        api_key=config.OPENROUTER_API_KEY,
        model=config.OPENROUTER_MODEL,
        base_url=config.OPENROUTER_BASE_URL,
        app_name=config.OPENROUTER_APP_NAME,
        http_referer=config.OPENROUTER_HTTP_REFERER,
    )
    print("✅ OpenRouter client ready.")
    return client
