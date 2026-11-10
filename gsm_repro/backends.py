from __future__ import annotations
import os
import re
from urllib.parse import urlsplit

class CallFailure(RuntimeError):
    def __init__(self, category, *, retryable=False, unknown=False, status_code=None, request_id=None,
                 provider_error_code=None, provider_error_message=None,
                 provider_error_type=None, provider_error_param=None):
        super().__init__(category)
        self.document = dict(category=category,retryable=retryable,unknown=unknown,
                             status_code=status_code,provider_request_id=request_id)
        self.document.update(provider_error_code=provider_error_code,
                             provider_error_message=provider_error_message,
                             provider_error_type=provider_error_type,
                             provider_error_param=provider_error_param)


def safe_error_text(value, secrets=(), *, limit=2000):
    """Keep bounded scalar diagnostics; never persist credentials or full response objects."""
    if value is None or not isinstance(value, (str, int, float, bool)):
        return None
    text = str(value)
    for secret in sorted((s for s in secrets if isinstance(s, str) and s), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)\bBearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_.-]+", "[REDACTED]", text)
    return text[:limit] + ("...[truncated]" if len(text) > limit else "")


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
            if len(key) < 2 or any(char.isspace() for char in key):
                raise ValueError("API key is incomplete or contains whitespace; re-enter the full key locally")
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Use an HTTPS endpoint without embedded credentials/query")
            from openai import OpenAI
            # No constructor default may add unrecorded retries.
            client = OpenAI(api_key=key,base_url=url,timeout=config["timeout"],max_retries=0)
        self.client = client
        self._secrets = (os.getenv(config["api_key_env"]), getattr(client, "api_key", None))
        if getattr(client,"max_retries",None) != 0:
            raise ValueError("Injected SDK client must have max_retries=0")
        self.endpoint_sha256 = digest(str(client.base_url).rstrip("/"))

    def complete(self, request):
        if os.getenv("GSM_REPRO_DISABLE_MODEL_CALLS") == "1":
            raise ValueError("Real model calls are disabled by process guard")
        import openai
        try:
            if request.get("stream"):
                return self._complete_stream(request)
            response = self.client.chat.completions.create(**request)
        except openai.APITimeoutError as exc:
            raise CallFailure("timeout",unknown=True) from exc
        except openai.APIConnectionError as exc:
            raise CallFailure("connection_error",retryable=True,unknown=True) from exc
        except openai.APIStatusError as exc:
            body = exc.body if isinstance(exc.body,dict) else {}
            # The SDK may unwrap {"error": ...}, losing a top-level request_id.
            try:
                response_body = exc.response.json()
            except ValueError:
                response_body = {}
            if not isinstance(response_body,dict):
                response_body = {}
            error = body.get("error",body)
            code = error.get("code","") if isinstance(error,dict) else ""
            details = error if isinstance(error,dict) else {}
            headers = exc.response.headers
            request_id = (getattr(exc,"request_id",None) or headers.get("x-dashscope-request-id")
                          or body.get("request_id") or body.get("requestId") or body.get("RequestId")
                          or response_body.get("request_id") or response_body.get("requestId")
                          or response_body.get("RequestId"))
            message = details.get("message") or (error if isinstance(error,str) else None)
            if message is None and isinstance(exc.body,str):
                message = exc.body
            diagnostics = {"provider_error_" + name: safe_error_text(value,self._secrets)
                           for name,value in dict(code=code or None,message=message,
                                                  type=details.get("type"),param=details.get("param")).items()}
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
                status_code=exc.status_code,request_id=safe_error_text(request_id,self._secrets),
                **diagnostics) from exc
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


    def _complete_stream(self, request):
        """Aggregate complete SSE replies; incomplete streams remain unconfirmed."""
        stream = self.client.chat.completions.create(**request)
        headers = getattr(getattr(stream, "response", None), "headers", {})
        provider_request_id = headers.get("x-request-id") or headers.get("x-dashscope-request-id")
        parts, reasoning_states = [], []
        model = fingerprint = finish = usage = None
        saw_choice = False
        try:
            for chunk in stream:
                model = getattr(chunk, "model", None) or model
                fingerprint = getattr(chunk, "system_fingerprint", None) or fingerprint
                usage = getattr(chunk, "usage", None) or usage
                choices = chunk.choices
                if len(choices) > 1:
                    raise CallFailure("invalid_response", unknown=True)
                if not choices:
                    continue
                choice = choices[0]
                if choice.index != 0:
                    raise CallFailure("invalid_response", unknown=True)
                saw_choice = True
                delta = choice.delta
                content = getattr(delta, "content", None)
                if content is not None:
                    if not isinstance(content, str):
                        raise CallFailure("invalid_response", unknown=True)
                    parts.append(content)
                values = delta.model_dump(exclude_unset=True)
                if "reasoning_content" in values:
                    value = values["reasoning_content"]
                    reasoning_states.append("null" if value is None else "empty" if value == "" else "nonempty")
                finish = choice.finish_reason or finish
        finally:
            stream.close()
        if not saw_choice or finish is None or not parts:
            raise CallFailure("invalid_response", unknown=True)
        reasoning = next((v for v in ("nonempty", "empty", "null") if v in reasoning_states), "missing")
        details = getattr(usage, "completion_tokens_details", None)
        return dict(content="".join(parts), model=model, finish_reason=finish,
            provider_request_id=provider_request_id, system_fingerprint=fingerprint, reasoning_state=reasoning,
            usage={key:getattr(usage,key,None) for key in ("prompt_tokens","completion_tokens","total_tokens")},
            completion_token_details=details.model_dump() if details is not None else None)
