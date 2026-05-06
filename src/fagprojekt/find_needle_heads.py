import re
from collections import defaultdict

#_______________VIBE CODED__________________
def find_best_average_needle_head(txt_path):
    # Regex pattern to extract layer, head, and attention values from each line
    pattern = re.compile(
        r"layer=\s*(\d+), head=\s*(\d+) \| "
        r"needle_attn=([0-9.]+) \| "
        r"max_attn=([0-9.]+) \| "
        r"mean_attn=([0-9.]+)")

    # Dictionary mapping (layer, head) -> list of needle_attn values across runs
    scores = defaultdict(list)

    # Read file line by line and extract matching entries
    with open(txt_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                layer = int(match.group(1))
                head = int(match.group(2))
                needle_attn = float(match.group(3))

                # Store needle attention for this head
                scores[(layer, head)].append(needle_attn)

    # Handle case where no valid lines were found
    if not scores:
        print("No needle heads found.")
        return None

    averages = []

    # Compute average needle attention per (layer, head)
    for (layer, head), values in scores.items():
        averages.append({
            "layer": layer,
            "head": head,
            "avg_needle_attention": sum(values) / len(values),  # mean over runs
            "num_occurrences": len(values),                     # how often this head appeared
            "values": values,                                  # raw values (optional debug)
        })

    # Sort primarily by how often the head appears,
    # and secondarily by average attention
    averages.sort(
        key=lambda x: (x["num_occurrences"], x["avg_needle_attention"]),
        reverse=True
    )

    # Best head = highest average attention
    best = averages[0]

    print("--- MOST CONSISTENT NEEDLE HEAD ---")
    print(
        f"layer={best['layer']}, head={best['head']} | "
        f"avg_needle_attn={best['avg_needle_attention']:.6f} | "
        f"seen={best['num_occurrences']} times")

    print("\n--- TOP CONSISTENT NEEDLE HEADS ---")
    # Print top 10 heads by average attention
    for r in averages[:10]:
        print(
            f"layer={r['layer']:2d}, head={r['head']:2d} | "
            f"avg_needle_attn={r['avg_needle_attention']:.6f} | "
            f"seen={r['num_occurrences']} times")

    return best, averages


# Run on your log file
best, averages = find_best_average_needle_head("logs/head_level_eval_100tokens.out")

#found: layer=22, head= 3 | avg_needle_attn=0.280908 | seen=7 times