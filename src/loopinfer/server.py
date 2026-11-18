import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenizers.decoders import DecodeStream

from .engine import Overloaded, Sampling

LOGGER = logging.getLogger(__name__)
SUPPORTED = {"model", "prompt", "max_tokens", "temperature", "top_p", "top_k", "seed", "stream", "n", "ignore_eos"}


def parse_request(payload, model_name):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    if unsupported := payload.keys() - SUPPORTED:
        raise ValueError(f"Unsupported parameters: {', '.join(sorted(unsupported))}")
    if payload.get("model", model_name) != model_name:
        raise ValueError(f"Model is not loaded: {payload.get('model')!r}")
    if payload.get("n", 1) != 1:
        raise ValueError("Only n=1 is supported")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) and not (
        isinstance(prompt, list) and all(type(token) is int for token in prompt)
    ):
        raise ValueError("prompt must be a string or an array of token IDs")
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise ValueError("stream must be a boolean")
    sample = {key: payload[key] for key in Sampling.__dataclass_fields__ if key in payload}
    if "ignore_eos" in sample and type(sample["ignore_eos"]) is not bool:
        raise ValueError("ignore_eos must be a boolean")
    return prompt, Sampling(**sample), stream


def completion(base, text, reason, **fields):
    choice = {"text": text, "index": 0, "logprobs": None, "finish_reason": reason, **fields}
    return {**base, "choices": [choice]}


def make_server(engine, host, port, *, timeout=120.0):
    if timeout <= 0:
        raise ValueError("Request timeout must be positive")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(timeout)

        def log_message(self, fmt, *args):
            LOGGER.debug(fmt, *args)

        def _json(self, status, data):
            blob = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def _error(self, status, message):
            error_type = "invalid_request_error" if 400 <= status < 500 else "server_error"
            self._json(status, {"error": {"message": message, "type": error_type}})

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "alive"})
            elif self.path == "/ready":
                self._json(200 if engine.ready else 503, {"ready": engine.ready})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": engine.loaded.name, "object": "model", "created": 0, "owned_by": "loopinfer"},
                ]})
            elif self.path == "/metrics":
                stats = engine.stats()
                blob = "".join(f"loopinfer_{key} {value}\n" for key, value in stats.items()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
            else:
                self._error(404, "Endpoint not found")

        def _events(self, handle, base):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            decoder = DecodeStream(skip_special_tokens=True)

            def send(value):
                self.wfile.write(f"data: {value}\n\n".encode("utf-8"))
                self.wfile.flush()

            try:
                for event in handle.consume(timeout):
                    delta = decoder.step(engine.tokenizer, event.token_id) or ""
                    data = completion(base, delta, event.finish_reason, loopinfer_token_id=event.token_id)
                    send(json.dumps(data, ensure_ascii=False))
            except (BrokenPipeError, ConnectionResetError):
                handle.cancel()
                return
            except Exception as exc:
                LOGGER.warning("Stream %s ended: %s", base["id"], exc)
                send(json.dumps({"error": {"message": "Generation failed", "type": "server_error"}}))
            send("[DONE]")

        def do_POST(self):
            if self.path != "/v1/completions":
                self._error(404, "Endpoint not found")
                return
            handle = None
            streaming_started = False
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2 * 1024**2:
                    raise ValueError("Content-Length must be between 1 byte and 2 MiB")
                payload = json.loads(self.rfile.read(size))
                prompt, sampling, stream = parse_request(payload, engine.loaded.name)
                handle = engine.submit(prompt, sampling, stream=stream)
                base = {
                    "id": f"cmpl-{uuid.uuid4().hex}", "created": int(time.time()),
                    "object": "text_completion", "model": engine.loaded.name,
                }
                if stream:
                    streaming_started = True
                    self._events(handle, base)
                    return
                result = handle.wait(timeout)
                data = completion(base, result.text, result.finish_reason)
                data["usage"] = {
                    "prompt_tokens": result.prompt_tokens, "completion_tokens": len(result.token_ids),
                    "total_tokens": result.prompt_tokens + len(result.token_ids),
                }
                self._json(200, data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, TypeError) as exc:
                if not streaming_started:
                    self._error(400, str(exc))
            except Overloaded as exc:
                if not streaming_started:
                    self._error(503, str(exc))
            except TimeoutError:
                if not streaming_started:
                    self._error(504, "Generation timed out")
            except Exception:
                LOGGER.exception("Generation failed")
                if not streaming_started:
                    self._error(500, "Generation failed")
            finally:
                if handle is not None and not handle.done.is_set():
                    handle.cancel()

    return ThreadingHTTPServer((host, port), Handler)
