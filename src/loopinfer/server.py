import json
import logging
import queue
import time
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenizers.decoders import DecodeStream

from .chat import render
from .engine import Overloaded, Sampling

LOGGER = logging.getLogger(__name__)
COMPLETION_FIELDS = {
    "model", "prompt", "max_tokens", "temperature", "top_p", "top_k", "seed", "stream", "n", "ignore_eos",
    "stop", "logprobs", "loop_depth", "exit_threshold", "stream_options",
}
CHAT_FIELDS = {
    "model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "seed",
    "stream", "n", "ignore_eos", "stop", "logprobs", "top_logprobs", "loop_depth", "exit_threshold",
}


def _stops(value):
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError("stop must be a string or an array of non-empty strings")
    return tuple(value)


def _choices(value, limit=8):
    if type(value) is not int or not 1 <= value <= limit:
        raise ValueError(f"n must be an integer between 1 and {limit}")
    return value


def _sampling(payload, *, max_tokens, logprobs):
    if "ignore_eos" in payload and type(payload["ignore_eos"]) is not bool:
        raise ValueError("ignore_eos must be a boolean")
    return Sampling(
        max_tokens=max_tokens, temperature=payload.get("temperature", 0.0), top_p=payload.get("top_p", 1.0),
        top_k=payload.get("top_k", 0), seed=payload.get("seed"), ignore_eos=payload.get("ignore_eos", False),
        stop=_stops(payload.get("stop")), logprobs=logprobs, loop_depth=payload.get("loop_depth"),
        exit_threshold=payload.get("exit_threshold"),
    )


def parse_completion(payload, engine):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    if unsupported := payload.keys() - COMPLETION_FIELDS:
        raise ValueError(f"Unsupported parameters: {', '.join(sorted(unsupported))}")
    if payload.get("model", engine.loaded.name) != engine.loaded.name:
        raise ValueError(f"Model is not loaded: {payload.get('model')!r}")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) and not (
        isinstance(prompt, list) and all(type(token) is int for token in prompt)
    ):
        raise ValueError("prompt must be a string or an array of token IDs")
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise ValueError("stream must be a boolean")
    logprobs = payload.get("logprobs")
    if logprobs is not None and (type(logprobs) is not int or not 0 <= logprobs <= 20):
        raise ValueError("logprobs must be an integer between 0 and 20")
    sampling = _sampling(payload, max_tokens=payload.get("max_tokens", 32), logprobs=logprobs)
    return {"prompt": prompt}, sampling, stream, _choices(payload.get("n", 1))


def parse_chat(payload, engine, template, default):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    if unsupported := payload.keys() - CHAT_FIELDS:
        raise ValueError(f"Unsupported parameters: {', '.join(sorted(unsupported))}")
    if payload.get("model", engine.loaded.name) != engine.loaded.name:
        raise ValueError(f"Model is not loaded: {payload.get('model')!r}")
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise ValueError("stream must be a boolean")
    wanted = payload.get("logprobs", False)
    if type(wanted) is not bool:
        raise ValueError("logprobs must be a boolean")
    top = payload.get("top_logprobs", 0)
    if type(top) is not int or not 0 <= top <= 20:
        raise ValueError("top_logprobs must be an integer between 0 and 20")
    max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens") or 32
    sampling = _sampling(payload, max_tokens=max_tokens, logprobs=top if wanted else None)
    tokens = engine.tokenizer
    bos = tokens.id_to_token(engine.loaded.bos) if engine.loaded.bos is not None else None
    prompt = render(
        payload.get("messages"), template, default=default, bos_token=bos or "",
        eos_token=" ".join(filter(None, (tokens.id_to_token(item) for item in engine.loaded.eos))),
    )
    return {"prompt": prompt}, sampling, stream, _choices(payload.get("n", 1))


def completion_logprobs(result, tokenizer):
    tokens, offsets, offset = [], [], 0
    for token in result.token_ids:
        piece = tokenizer.decode([token])
        tokens.append(piece)
        offsets.append(offset)
        offset += len(piece)
    return {
        "tokens": tokens, "text_offset": offsets,
        "token_logprobs": [float(item[0]) for item in result.logprobs],
        "top_logprobs": [
            {tokenizer.decode([token]): float(value) for token, value in item[1]} for item in result.logprobs
        ],
    }


def chat_logprobs(result, tokenizer):
    return {"content": [
        {
            "token": tokenizer.decode([token]), "logprob": float(logprob),
            "top_logprobs": [
                {"token": tokenizer.decode([token_id]), "logprob": float(value)} for token_id, value in top
            ],
        }
        for token, (logprob, top) in zip(result.token_ids, result.logprobs)
    ]}


def make_server(engine, host, port, *, timeout=120.0, chat_template=None, default_chat_template=None):
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

        def _start_events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def _send(self, data):
            self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        def _chunk(self, kind, base, index, delta, reason, logprob=None, token_id=None):
            if token_id is not None and self.headers.get("X-LoopInfer-Token-Ids") == "1":
                base = {**base, "token_ids": [token_id]}
            if kind == "chat":
                choice = {"index": index, "delta": delta if isinstance(delta, dict) else {"content": delta},
                          "finish_reason": reason}
                if logprob is not None:
                    choice["logprobs"] = {"content": [logprob]}
                return {**base, "object": "chat.completion.chunk", "choices": [choice]}
            return {**base, "object": "text_completion",
                    "choices": [{"index": index, "text": delta if isinstance(delta, str) else "",
                                 "logprobs": None, "finish_reason": reason}]}

        def _events(self, entries, kind, base, include_usage=False):
            """entries: [(handle, choice_index, sampling)]; streams are interleaved."""
            self._start_events()
            decoders = [DecodeStream(skip_special_tokens=True) for _ in entries]
            seen = [""] * len(entries)
            emitted = [0] * len(entries)
            deadline = time.monotonic() + timeout
            holdback = [max((len(stop) for stop in sampling.stop), default=1) - 1 for _, _, sampling in entries]
            if kind == "chat":
                for _, index, _ in entries:
                    self._send(self._chunk(kind, base, index, {"role": "assistant", "content": ""}, None))
            live = set(range(len(entries)))
            finished_sent = [False] * len(entries)
            try:
                while live:
                    if time.monotonic() >= deadline:
                        self._send({"error": {"message": "Generation timed out", "type": "server_error"}})
                        break
                    idle = True
                    for slot in sorted(live):
                        handle, index, _ = entries[slot]
                        try:
                            event = handle.events.get_nowait()
                        except queue.Empty:
                            if not handle.done.is_set():
                                continue
                            finished = handle.result
                            if finished is None:
                                self._send({"error": {"message": str(handle.error), "type": "server_error"}})
                                live.discard(slot)
                                idle = False
                                continue
                            tail = finished.text[emitted[slot]:]
                            if tail or not finished_sent[slot]:
                                self._send(self._chunk(
                                    kind, base, index, {"content": tail} if kind == "chat" else tail,
                                    finished.finish_reason,
                                ))
                                emitted[slot] = len(finished.text)
                            live.discard(slot)
                            idle = False
                            continue
                        idle = False
                        seen[slot] += decoders[slot].step(engine.tokenizer, event.token_id) or ""
                        limit = len(seen[slot]) - holdback[slot]
                        piece = ""
                        if limit > emitted[slot]:
                            piece = seen[slot][emitted[slot]:limit]
                            emitted[slot] = limit
                        logprob = None
                        if event.logprob is not None:
                            logprob = chat_logprobs_one(engine.tokenizer, event.token_id, event.logprob, event.top)
                        if event.finish_reason:
                            finished_sent[slot] = True
                        self._send(self._chunk(
                            kind, base, index, {"content": piece} if kind == "chat" else piece,
                            event.finish_reason, logprob, event.token_id,
                        ))
                    if idle:
                        time.sleep(0.002)
            except (BrokenPipeError, ConnectionResetError):
                for handle, _, _ in entries:
                    handle.cancel()
                return
            finally:
                for slot in live:
                    entries[slot][0].cancel()
            if include_usage and all(handle.result is not None for handle, _, _ in entries):
                prompt_tokens = sum(handle.result.prompt_tokens for handle, _, _ in entries)
                completion_tokens = sum(len(handle.result.token_ids) for handle, _, _ in entries)
                self._send({**self._chunk(kind, base, 0, "", None), "choices": [], "usage": {
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _submit(self, prompt, sampling, stream, count):
            handles = []
            for index in range(count):
                sample = sampling
                if count > 1 and sampling.seed is not None:
                    sample = replace(sampling, seed=(sampling.seed + index) % 2**64)
                try:
                    handles.append(engine.submit(prompt, sample, stream=stream))
                except Exception:
                    for handle in handles:
                        handle.cancel()
                    raise
            return handles

        def do_POST(self):
            kind = {"/v1/completions": "completion", "/v1/chat/completions": "chat"}.get(self.path)
            if kind is None:
                self._error(404, "Endpoint not found")
                return
            handles, streaming_started = [], False
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2 * 1024**2:
                    raise ValueError("Content-Length must be between 1 byte and 2 MiB")
                payload = json.loads(self.rfile.read(size))
                chat = kind == "chat"
                source, sampling, stream, count = (
                    parse_chat(payload, engine, chat_template, default_chat_template) if chat
                    else parse_completion(payload, engine)
                )
                stream_options = payload.get("stream_options") or {}
                if (not isinstance(stream_options, dict) or stream_options.keys() - {"include_usage"}
                        or type(stream_options.get("include_usage", False)) is not bool
                        or (stream_options and not stream)):
                    raise ValueError("stream_options.include_usage requires a streaming request")
                handles = self._submit(source["prompt"], sampling, stream, count)
                base = {
                    "id": f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}", "created": int(time.time()),
                    "model": engine.loaded.name,
                }
                if stream:
                    streaming_started = True
                    self._events([(handle, index, sampling) for index, handle in enumerate(handles)],
                                 "chat" if chat else "completion", base, stream_options.get("include_usage", False))
                    return
                results = [handle.wait(timeout) for handle in handles]
                usage = {
                    "prompt_tokens": sum(result.prompt_tokens for result in results),
                    "completion_tokens": sum(len(result.token_ids) for result in results),
                }
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                if chat:
                    choices = [{
                        "index": index, "message": {"role": "assistant", "content": result.text},
                        "finish_reason": result.finish_reason,
                        "logprobs": chat_logprobs(result, engine.tokenizer) if result.logprobs else None,
                    } for index, result in enumerate(results)]
                    self._json(200, {**base, "object": "chat.completion", "choices": choices, "usage": usage})
                else:
                    choices = [{
                        "text": result.text, "index": index, "finish_reason": result.finish_reason,
                        "logprobs": completion_logprobs(result, engine.tokenizer) if result.logprobs else None,
                    } for index, result in enumerate(results)]
                    self._json(200, {**base, "object": "text_completion", "choices": choices, "usage": usage})
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
                for handle in handles:
                    if not handle.done.is_set():
                        handle.cancel()

    return ThreadingHTTPServer((host, port), Handler)


def chat_logprobs_one(tokenizer, token, logprob, top):
    return {
        "token": tokenizer.decode([token]), "logprob": float(logprob),
        "top_logprobs": [
            {"token": tokenizer.decode([token_id]), "logprob": float(value)} for token_id, value in top
        ],
    }
