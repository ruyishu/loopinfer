"""Render chat messages into a prompt with the checkpoint's Jinja template."""

DEFAULT_TEMPLATES = {
    "chatml": (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    ),
}
ROLES = {"system", "user", "assistant", "tool"}


def normalise(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object")
        role = message.get("role")
        if role not in ROLES:
            raise ValueError(f"Unsupported message role {role!r}")
        content = message.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                    raise ValueError("Only text content parts are supported")
                texts.append(part["text"])
            content = "".join(texts)
        if not isinstance(content, str):
            raise ValueError("Message content must be a string or an array of text parts")
        cleaned.append({"role": role, "content": content})
    return cleaned


def render(messages, template, *, default=None, bos_token="", eos_token="", add_generation_prompt=True):
    if template is None and default:
        template = DEFAULT_TEMPLATES.get(default)
        if template is None:
            raise ValueError(f"Unknown default chat template {default!r}")
    if template is None:
        raise ValueError(
            "This checkpoint ships no chat template; start the server with --chat-template PATH "
            "or --default-chat-template chatml"
        )
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as exc:
        raise ValueError("Chat templates require jinja2, which is not installed") from exc
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message):
        raise ValueError(message)

    compiled = environment.from_string(template)
    return compiled.render(
        messages=normalise(messages), add_generation_prompt=add_generation_prompt,
        bos_token=bos_token, eos_token=eos_token, raise_exception=raise_exception,
    )
