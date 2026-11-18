import threading

import httpx
from openai import OpenAI

from loopinfer import Engine
from loopinfer.bench import make_trace, replay, run
from loopinfer.server import make_server


def serve(engine, **options):
    server = make_server(engine, "127.0.0.1", 0, **options)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, "http://127.0.0.1:" + str(server.server_address[1])


def test_openai_and_benchmark(checkpoint):
    with Engine(str(checkpoint), device="cpu", context=32, prefill_chunk=3, max_batch=4) as engine:
        server, thread, root = serve(engine, default_chat_template="chatml")
        try:
            payload = {"model": engine.loaded.name, "prompt": "a b", "max_tokens": 4, "ignore_eos": True}
            result = httpx.post(root + "/v1/completions", json=payload)
            assert result.status_code == 200
            assert result.json()["usage"]["completion_tokens"] == 4
            text = result.json()["choices"][0]["text"]

            scored = httpx.post(root + "/v1/completions", json={**payload, "logprobs": 2})
            assert scored.status_code == 200
            logprobs = scored.json()["choices"][0]["logprobs"]
            assert len(logprobs["tokens"]) == len(logprobs["top_logprobs"]) == 4
            assert all(logprob <= 0 for logprob in logprobs["token_logprobs"])

            stopped = httpx.post(root + "/v1/completions", json={**payload, "stop": ["zz"]})
            assert stopped.status_code == 200
            assert "zz" not in stopped.json()["choices"][0]["text"]

            chat = httpx.post(root + "/v1/chat/completions", json={
                "model": engine.loaded.name, "messages": [{"role": "user", "content": "a b"}],
                "max_tokens": 3, "ignore_eos": True, "logprobs": True, "top_logprobs": 1,
            })
            assert chat.status_code == 200
            body = chat.json()
            assert body["object"] == "chat.completion"
            assert body["choices"][0]["message"]["role"] == "assistant"
            assert len(body["choices"][0]["logprobs"]["content"]) == 3

            both = httpx.post(root + "/v1/completions", json={**payload, "n": 2})
            assert both.status_code == 200
            assert len(both.json()["choices"]) == 2

            with OpenAI(base_url=root + "/v1", api_key="local") as client:
                reply = client.completions.create(**{
                    k: v for k, v in payload.items() if k != "ignore_eos"
                }, stream=True, extra_body={"ignore_eos": True})
                chunks = list(reply)
                assert len(chunks) == 4
                assert chunks[-1].choices[0].finish_reason == "length"
                assert "".join(chunk.choices[0].text for chunk in chunks) == text

            unsupported = httpx.post(root + "/v1/completions", json={**payload, "best_of": 2})
            assert unsupported.status_code == 400

            report = run(root, None, 8, 4, 2, 3, token_id=3, warmup=1)
            assert report["completed"] == 3
            assert report["errors"] == []
            assert report["tpot_p95_ms"] is not None
            trace = make_trace("long-prefill", 2, input_tokens=8, request_rate=50, vocab_size=10)
            report, events = replay(root, None, trace, 2, warmup=0)
            assert report["completed"] == 2
            assert report["tpot_p95_ms"] is None
            assert all(row["output_tokens"] == 1 for row in events)
            metrics = httpx.get(root + "/metrics").text
            assert "loopinfer_kv_pages_used 0" in metrics
            assert "loopinfer_step_ms_forward" in metrics
        finally:
            server.shutdown()
            thread.join()


def test_chat_needs_a_template(checkpoint):
    with Engine(str(checkpoint), device="cpu", context=16, page_size=2) as engine:
        server, thread, root = serve(engine)
        try:
            reply = httpx.post(root + "/v1/chat/completions", json={
                "model": engine.loaded.name, "messages": [{"role": "user", "content": "a b"}],
            })
            assert reply.status_code == 400
            assert "--chat-template" in reply.json()["error"]["message"]
        finally:
            server.shutdown()
            thread.join()
