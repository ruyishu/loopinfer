import threading

import httpx
from openai import OpenAI

from loopinfer import Engine
from loopinfer.bench import run
from loopinfer.server import make_server


def test_openai_and_benchmark(checkpoint):
    with Engine(str(checkpoint), device="cpu", context=32, prefill_chunk=3, max_batch=4) as engine:
        with make_server(engine, "127.0.0.1", 0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            root = "http://127.0.0.1:" + str(server.server_address[1])
            try:
                payload = {"model": engine.loaded.name, "prompt": "a b", "max_tokens": 4, "ignore_eos": True}
                result = httpx.post(root + "/v1/completions", json=payload)
                assert result.status_code == 200
                assert result.json()["usage"]["completion_tokens"] == 4
                with OpenAI(base_url=root + "/v1", api_key="local") as client:
                    reply = client.completions.create(**{
                        k: v for k, v in payload.items() if k != "ignore_eos"
                    }, stream=True, extra_body={"ignore_eos": True})
                    chunks = list(reply)
                    assert len(chunks) == 4
                    assert chunks[-1].choices[0].finish_reason == "length"
                    assert "".join(chunk.choices[0].text for chunk in chunks) == result.json()["choices"][0]["text"]
                wrong = httpx.post(root + "/v1/completions", json={**payload, "stop": ["x"]})
                assert wrong.status_code == 400
                report = run(root, None, 8, 4, 2, 3, token_id=3, warmup=1)
                assert report["completed"] == 3
                assert report["errors"] == []
                assert "loopinfer_kv_pages_used 0" in httpx.get(root + "/metrics").text
            finally:
                server.shutdown()
                thread.join()
