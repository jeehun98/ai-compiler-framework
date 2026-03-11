from semantic import build_attention_semantic_graph
from passes import run_attention_pattern_pass, run_streaming_lowering_pass
from mcir.printer import dump_module
from mcir.validate import validate_module


def main() -> None:
    graph = build_attention_semantic_graph()

    module = run_attention_pattern_pass(graph)
    module = run_streaming_lowering_pass(module)

    validate_module(module)

    print(dump_module(module))


if __name__ == "__main__":
    main()