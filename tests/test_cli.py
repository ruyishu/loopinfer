from loopinfer.cli import parser


def test_cli_accepts_serve_options():
    options = parser().parse_args([
        "serve", "checkpoint", "--attention", "triton", "--graph-budget", "8",
        "--chat-template", "template.jinja", "--default-chat-template", "chatml",
        "--cuda-graph", "--context", "4096",
    ])
    assert options.attention == "triton"
    assert options.graph_budget == 8
    assert options.default_chat_template == "chatml"
    assert str(options.chat_template).endswith("template.jinja")


def test_cli_accepts_generate_and_bench():
    options = parser().parse_args(["generate", "checkpoint", "--prompt", "hi", "--no-cuda-graph"])
    assert options.cuda_graph is False
    assert parser().parse_args(["bench", "--concurrency", "2"]).concurrency == 2
