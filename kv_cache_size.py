import argparse


def kv_cache_size(n_layers, n_heads, d_head, precision):
    # per token, returns in bits
    return n_layers * n_heads * d_head * precision * 4


def get_attention_params(model, attention):
    """
    Define KV sizes for different attention mechanisms per model.
    """
    # https://blogs.vmware.com/cloud-foundation/2024/09/25/llm-inference-sizing-and-performance-guidance/
    model_params = {
        "GPT-3": {"MHA": (96, 64), "GQA": (48, 64), "MLA": (32, 64)},
        "LLaMA": {"MHA": (80, 128), "GQA": (40, 128), "MLA": (24, 128)},
        #   
        "Falcon": {"MHA": (64, 128), "GQA": (32, 128), "MLA": (16, 128)},
    }

    if model in model_params and attention in model_params[model]:
        return model_params[model][attention]
    else:
        raise ValueError("Model or attention mechanism not found.")


def generate_kv_table():
    """
    Generate a formatted table of model sizes vs. KV sizes per token.
    """
    models = ["GPT-3", "LLaMA", "Falcon"]
    attentions = ["MHA", "GQA", "MLA"]
    batch_size = 1
    seq_length = 1
    n_layers = 24
    precision = 2  # FP16 - 2 bytes

    table_data = []

    for model in models:
        for attention in attentions:
            n_heads, d_head = get_attention_params(model, attention)
            kv_size = kv_cache_size(
                batch_size, seq_length, n_layers, n_heads, d_head, precision
            )
            table_data.append([model, attention, kv_size])

    header = "| Model  | Attention | KV Size per Token (bytes) |"
    separator = "|--------|-----------|--------------------------|"
    rows = [f"| {m} | {a} | {s} |" for m, a, s in table_data]

    print("\n".join([header, separator] + rows))


if __name__ == "__main__":
    generate_kv_table()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., GPT-3, LLaMA, Falcon)",
    )
    parser.add_argument(
        "--attention",
        type=str,
        required=True,
        choices=["MHA", "GQA", "MLA"],
        help="Attention mechanism",
    )
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument(
        "--seq_length", type=int, required=True, help="Total sequence length"
    )
    parser.add_argument(
        "--n_layers", type=int, required=True, help="Number of attention layers"
    )
    parser.add_argument(
        "--precision",
        type=int,
        required=True,
        help="Precision (e.g., 2 for FP16, 4 for FP32)",
    )

    args = parser.parse_args()

    n_heads, d_head = get_attention_params(args.model, args.attention)
    kv_size = kv_cache_size(
        args.batch_size, args.seq_length, args.n_layers, n_heads, d_head, args.precision
    )

    print(f"KV Cache Size for {args.model} using {args.attention}: {kv_size} bytes")

    generate_kv_table()
