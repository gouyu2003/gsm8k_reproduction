from __future__ import annotations
import os
from urllib.parse import urlsplit

class CallFailure(RuntimeError):
    def __init__(self, category, *, retryable=False, unknown=False, status_code=None, request_id=None):
        super().__init__(category)
        self.document = dict(category=category,retryable=retryable,unknown=unknown,
                             status_code=status_code,provider_request_id=request_id)


class FakeBackend:
    kind = "fake"
    endpoint_sha256 = None
    def __init__(self):
        self.requests = []

    def complete(self, request):
        from copy import deepcopy
        self.requests.append(deepcopy(request))
        return dict(content="Offline fixed fake reply; no model was called. \\boxed{42}",
            model="offline-fake", finish_reason="stop", provider_request_id=None, system_fingerprint=None,
            reasoning_state="missing", usage=dict(prompt_tokens=None,completion_tokens=None,total_tokens=None))


class OpenAICompatibleBackend:
    kind = "openai_compatible"
    def __init__(self, config, *, allow_model_calls=False, client=None):
        if not allow_model_calls or os.getenv("GSM_REPRO_DISABLE_MODEL_CALLS") == "1":
            raise ValueError("Real model calls are disabled")
        from .common import digest
        if client is None:
            key, url = os.getenv(config["api_key_env"]), os.getenv(config["base_url_env"])
            if not key or not url:
                raise ValueError("Set the dedicated API key/base URL environment variables; no .env is loaded")
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Use an HTTPS endpoint without embedded credentials/query")
            from openai import OpenAI
            # No constructor default may add unrecorded retries.
            client = OpenAI(api_key=key,base_url=url,timeout=config["timeout"],max_retries=0)
        self.client = client
        if getattr(client,"max_retries",None) != 0:
            raise ValueError("Injected SDK client must have max_retries=0")
        self.endpoint_sha256 = digest(str(client.base_url).rstrip("/"))

    def complete(self, request):
        if os.getenv("GSM_REPRO_DISABLE_MODEL_CALLS") == "1":
            raise ValueError("Real model calls are disabled by process guard")
        import openai
        try:
            response = self.client.chat.completions.create(**request)
        except openai.APITimeoutError as exc:
            raise CallFailure("timeout",unknown=True) from exc
        except openai.APIConnectionError as exc:
            raise CallFailure("connection_error",retryable=True,unknown=True) from exc
        except openai.APIStatusError as exc:
            body = exc.body if isinstance(exc.body,dict) else {}
            error = body.get("error",body)
            code = error.get("code","") if isinstance(error,dict) else ""
            blocked = any(word in str(code).lower() for word in
                ("inspection","content_filter","safety","quota","balance","payment","invalid"))
            if any(word in str(code).lower() for word in ("quota","balance","payment")):
                category = "quota_error"
            elif any(word in str(code).lower() for word in ("inspection","content_filter","safety")):
                category = "provider_rejected"
            elif blocked or exc.status_code in (400,401,403,404,422):
                category = "configuration_error"
            else:
                category = "http_error"
            raise CallFailure(category,
                retryable=category=="http_error" and (exc.status_code in (408,429) or exc.status_code >= 500),
                status_code=exc.status_code,request_id=getattr(exc,"request_id",None)) from exc
        if len(response.choices) != 1 or not isinstance(response.choices[0].message.content,str):
            raise CallFailure("invalid_response",unknown=True)
        choice = response.choices[0]
        message = choice.message.model_dump()
        if "reasoning_content" not in message:
            reasoning = "missing"
        else:
            value = message["reasoning_content"]
            reasoning = "null" if value is None else "empty" if value == "" else "nonempty"
        usage = response.usage
        return dict(content=choice.message.content,model=response.model,finish_reason=choice.finish_reason,
            provider_request_id=getattr(response,"_request_id",None),
            system_fingerprint=getattr(response,"system_fingerprint",None),reasoning_state=reasoning,
            usage={key:getattr(usage,key,None) for key in ("prompt_tokens","completion_tokens","total_tokens")},
            completion_token_details=getattr(usage,"completion_tokens_details",None).model_dump()
                if getattr(usage,"completion_tokens_details",None) is not None else None)
