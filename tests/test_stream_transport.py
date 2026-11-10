import copy, json, os, tempfile, unittest
from unittest.mock import patch
import httpx, openai
from gsm_repro.backends import OpenAICompatibleBackend, CallFailure
from gsm_repro.budget import Budget
from gsm_repro.config import DEFAULT, request, validate

def event(delta=None, finish=None, usage=None, choices=True):
    d={"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"qwen3.5-27b", "choices": ([{"index":0,"delta":delta or {},"finish_reason":finish}] if choices else [])}
    if usage is not None:d["usage"]=usage
    return "data: " + json.dumps(d) + "\n\n"

class StreamTransportTests(unittest.TestCase):
    def setUp(self):
        self.cfg=copy.deepcopy(DEFAULT); self.cfg["stream"]=True
        self.cfg["pricing"]={"confirmed":True,"input_per_million_cny":"0.6","output_per_million_cny":"4.8","input_token_reserve":16384,"source":"offline","verified_at":"2026-11-10"}
        self.body=request(self.cfg,[{"role":"user","content":"q"}],"single:0:q:0:0")
    def execute(self, content):
        seen=[]
        def respond(req):
            seen.append(json.loads(req.content)); return httpx.Response(200,headers={"content-type":"text/event-stream","x-dashscope-request-id":"mock-id"},content=content.encode())
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            client=openai.OpenAI(api_key="offline",base_url="https://offline.invalid/v1",max_retries=0,http_client=http)
            with patch.dict(os.environ,{"GSM_REPRO_DISABLE_MODEL_CALLS":"0"}):
                backend=OpenAICompatibleBackend(self.cfg,allow_model_calls=True,client=client)
                response=backend.complete(self.body)
        return seen,response
    def test_request_stream_is_optional_and_usage_is_requested(self):
        old=copy.deepcopy(DEFAULT);validate(old)
        self.assertFalse(request(old,[],"x")["stream"])
        validate(self.cfg); self.assertTrue(self.body["stream"]);self.assertEqual(self.body["stream_options"],{"include_usage":True})
        self.cfg["stream"]="true"
        with self.assertRaises(ValueError):validate(self.cfg)
    def test_fragments_finish_usage_and_reasoning(self):
        s=event({"role":"assistant","reasoning_content":""})+event({"content":"ans "})+event({"content":"42"},"stop")+event(usage={"prompt_tokens":10,"completion_tokens":3,"total_tokens":13},choices=False)+"data: [DONE]\n\n"
        seen,r=self.execute(s);self.assertTrue(seen[0]["stream"]);self.assertEqual(r["content"],"ans 42");self.assertEqual(r["finish_reason"],"stop");self.assertEqual(r["reasoning_state"],"empty");self.assertEqual(r["usage"]["total_tokens"],13);self.assertEqual(r["provider_request_id"],"mock-id")
    def test_reasoning_and_length_are_retained(self):
        s=event({"reasoning_content":"hidden","content":"x"},"length")+event(usage={"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},choices=False)+"data: [DONE]\n\n"
        _,r=self.execute(s);self.assertEqual(r["reasoning_state"],"nonempty");self.assertEqual(r["finish_reason"],"length")
    def test_incomplete_stream_is_unknown(self):
        with self.assertRaises(CallFailure) as ctx:self.execute(event({"content":"partial"})+"data: [DONE]\n\n")
        self.assertTrue(ctx.exception.document["unknown"])
    def test_missing_usage_cannot_settle_budget(self):
        _,r=self.execute(event({"content":"x"},"stop")+"data: [DONE]\n\n")
        with tempfile.TemporaryDirectory() as d:
            b=Budget(d,self.cfg);b.reserve("id",self.body);self.assertFalse(b.settle("id",r))

if __name__=="__main__":unittest.main()
