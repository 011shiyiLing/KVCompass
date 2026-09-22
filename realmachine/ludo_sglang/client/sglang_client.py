"""
SGLang HTTP client — uses the *native* ``/generate`` endpoint, not the
OpenAI-compatible ``/v1/chat/completions``.

Why not OpenAI-compat:
  This SGLang version's ``/v1/chat/completions`` returns
  ``usage.prompt_tokens_details = null`` — i.e. it does not surface the
  RadixCache hit count. Without that field every request looks like a
  cold miss in our CSVs, and we cannot tell whether routing-induced
  cache reuse actually translates into prefill skipping.

The native ``/generate`` endpoint instead returns ``meta_info`` with
``cached_tokens`` populated per request. The trade-off is that the
caller must apply the model's chat template manually before sending
(SGLang's OpenAI shim does this internally, but ``/generate`` takes
raw text / tokens).

Per-request timing:
  - ``ttft_ms`` = wall-clock between the POST and the first SSE chunk
    that carries non-empty ``text``.
  - ``e2e_ms``  = wall-clock between the POST and stream close.
  Both are measured with ``time.monotonic()``.

NOTE: ``ttft_ms`` / ``e2e_ms`` measure the SGLang side only. They do
NOT include any ``await asyncio.sleep(overhead)`` injected by the
router's scheme (Chord, Kademlia, CD), which happens before the client
is called. The plotter combines these two to produce
``effective_ttft_ms``.
"""
import json
import time

import aiohttp


async def async_send_request(
    worker_url: str,
    formatted_prompt: str,
    max_new_tokens: int,
    http_timeout: float = 600.0,
) -> dict:
    """
    POST to ``<worker_url>/generate`` with a chat-templated prompt
    string, consume the SSE stream, return a result dict.

    Parameters
    ----------
    worker_url       : base URL of the SGLang worker (no trailing slash)
    formatted_prompt : already-templated prompt text — typically the
                       result of ``tokenizer.apply_chat_template(...,
                       tokenize=False, add_generation_prompt=True)``.
    max_new_tokens   : sampling budget.
    http_timeout     : total session timeout (covers slow tail).

    Returns
    -------
    dict with keys:
      text, ttft_ms, e2e_ms, prompt_tokens, completion_tokens,
      cached_tokens
    """
    payload = {
        "text": formatted_prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature":    0.0,
            "ignore_eos":     False,
        },
        "stream": True,
    }

    output_text       = ""
    cached_tokens     = 0
    prompt_tokens     = 0
    completion_tokens = 0
    ttft_ms: float | None = None

    timeout = aiohttp.ClientTimeout(total=http_timeout)
    t0 = time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{worker_url}/generate",
            json=payload,
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"SGLang returned {response.status}: {body}")

            # SGLang /generate streaming format: each SSE event is a
            # "data: {...}\n\n" frame whose JSON contains ``text``
            # (cumulative output so far) and ``meta_info`` (running
            # token counts, including cached_tokens). The final frame
            # additionally carries ``meta_info.finish_reason``.
            prev_text = ""
            async for line_bytes in response.content:
                line = line_bytes.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Update token counts from meta_info on every frame —
                # the final frame is authoritative but intermediate
                # frames are also correct.
                meta = data.get("meta_info") or {}
                if meta:
                    prompt_tokens     = meta.get("prompt_tokens",     prompt_tokens)
                    completion_tokens = meta.get("completion_tokens", completion_tokens)
                    cached_tokens     = meta.get("cached_tokens",     cached_tokens)

                # SGLang sends cumulative text — extract just the new
                # delta to detect "first token observed".
                cur_text = data.get("text", "") or ""
                if len(cur_text) > len(prev_text):
                    delta = cur_text[len(prev_text):]
                    prev_text = cur_text
                    if delta and ttft_ms is None:
                        ttft_ms = (time.monotonic() - t0) * 1000.0
                    output_text = cur_text
    e2e_ms = (time.monotonic() - t0) * 1000.0

    return {
        "text":              output_text,
        "ttft_ms":           ttft_ms if ttft_ms is not None else e2e_ms,
        "e2e_ms":            e2e_ms,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens":     cached_tokens,
    }
