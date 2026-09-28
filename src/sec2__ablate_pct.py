"""Script from Henry: Ablate increasing percentages of OCR-ranked attention heads.

Instead of ablating the top-k heads, we can also ablate the top-p% of heads. 
This script runs a spacing of percentages and also samples the same number of 
random heads as a baseline, selected from all layers, the last 50% of layers, 
and the last 30% of layers. 

In order to run ablations, you must first run src/sec2__cache_activations.py to get
mean activations to use for mean-ablations, as well as sec2__score_heads.py to get 
scores for each head. 
"""
import math 
import argparse
import json
import random
import sys
import torch
from pathlib import Path

ROOT = Path("")

MODELS = (
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
)
PERCENTAGES = (1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
SCOPES = ("all", "last50", "last30")
PROMPT = "Transcribe the word in this image."
PREFILL = "Word:"

HEAD_MEAN_PATHS = {
    "Qwen3-VL-2B-Instruct": Path(
        "cache/Qwen3-VL-2B-Instruct/"
        "images__score_ocr_heads__Qwen3-VL-2B-Instruct_seed177/examples1000.pt"
    ),
    "Qwen3-VL-8B-Instruct": Path(
        "cache/Qwen3-VL-8B-Instruct/"
        "images__score_ocr_heads__Qwen3-VL-8B-Instruct_seed177/examples1000.pt"
    ),
}

class ImageDirectory:
    """Small dataset wrapper for local background images."""

    def __init__(self, root):
        self.paths = sorted(
            path
            for path in root.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not self.paths:
            raise ValueError(f"No images found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            return {"image": image.convert("RGB").copy()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument(
        "--score-path",
        type=Path,
        help="Default: results/find_ocr_heads/<model>/examples1500_seed177_final.pt",
    )
    parser.add_argument(
        "--head-means",
        type=Path,
        help="Default: the matching file in cache/<model>/.",
    )
    parser.add_argument(
        "--percentages", type=float, nargs="+", default=list(PERCENTAGES)
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=SCOPES,
        default=list(SCOPES),
        help="Layer pools for the random controls; OCR ranking is always global.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=10,
        help="Number of independent random-head draws per (scope, percentage).",
    )
    parser.add_argument(
        "--conditions",
        choices=("all", "ocr", "random"),
        default="all",
        help="Run all conditions, only baseline/OCR, or only random controls.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conditions already present in results.csv (resume a sweep).",
    )
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=177)
    parser.add_argument("--word-list", type=Path, default=Path("/usr/share/dict/words"))
    parser.add_argument(
        "--background-dir",
        type=Path,
        help="Optional local image directory instead of timm/mini-imagenet.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/ablate_ocr_heads")
    )
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conditions without loading the model or dataset.",
    )
    return parser.parse_args()


def resolve_path(path):
    if path is None or path.is_absolute():
        return path
    return ROOT / path


def model_name(model_id):
    return model_id.split("/")[-1]


# Keys that may grow between runs without invalidating results already on disk:
# each condition is seeded independently, so adding percentages, scopes, or
# random trials only appends new conditions and never changes existing ones.
EXTENSIBLE_KEYS = ("percentages", "random_scopes", "random_trials")


def validate_run_config(path, config):
    if path.exists():
        existing = json.loads(path.read_text())
        fixed_existing = {
            key: value
            for key, value in existing.items()
            if key not in EXTENSIBLE_KEYS
        }
        fixed_new = {
            key: value for key, value in config.items() if key not in EXTENSIBLE_KEYS
        }
        if fixed_existing != fixed_new:
            raise ValueError(
                f"{path.parent} contains results from a different configuration; "
                "use another --output-dir"
            )
        merged = dict(config)
        merged["percentages"] = sorted(
            set(existing.get("percentages", [])) | set(config["percentages"])
        )
        merged["random_scopes"] = list(
            dict.fromkeys(
                list(existing.get("random_scopes", [])) + config["random_scopes"]
            )
        )
        merged["random_trials"] = max(
            existing.get("random_trials", 0), config["random_trials"]
        )
        if merged != existing:
            path.write_text(json.dumps(merged, indent=2) + "\n")
    elif (path.parent / "results.csv").exists():
        raise ValueError(
            f"{path.parent} has results without a run_config.json; "
            "use another --output-dir"
        )
    else:
        path.write_text(json.dumps(config, indent=2) + "\n")


def load_head_scores(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(saved, (tuple, list)):
        scores = saved[0]
    elif isinstance(saved, dict):
        scores = saved.get("mean_scores", saved.get("scores"))
    else:
        scores = saved
    if not torch.is_tensor(scores) or scores.ndim != 2:
        raise ValueError(f"Expected a [layers, heads] tensor in {path}")
    return scores.float()


def load_head_means(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    means = saved.get("means") if isinstance(saved, dict) else saved
    if not torch.is_tensor(means) or means.ndim != 2:
        raise ValueError(f"Expected a [layers, heads * head_dim] tensor in {path}")
    return means


def layers_in_scope(scope, n_layers):
    first_layer = {
        "all": 0,
        "last50": n_layers // 2,
        "last30": n_layers - math.floor(n_layers * 0.3),
    }[scope]
    return range(first_layer, n_layers)


def build_conditions(scores, percentages, scopes, random_trials, seed):
    n_layers, n_heads = scores.shape
    all_heads = [
        (layer, head) for layer in range(n_layers) for head in range(n_heads)
    ]
    ranked = sorted(
        all_heads, key=lambda head: float(scores[head]), reverse=True
    )
    random_pools = {
        scope: [
            (layer, head)
            for layer in layers_in_scope(scope, n_layers)
            for head in range(n_heads)
        ]
        for scope in scopes
    }
    conditions = [
        {
            "name": "baseline",
            "scope": "all",
            "percent": 0,
            "kind": "baseline",
            "trial": None,
            "heads": [],
        }
    ]

    for percent in percentages:
        # int truncates numbers, so +0.5 makes it equiv to rounding 
        n_ablate = max(
            1, min(len(all_heads), int(len(all_heads) * percent / 100 + 0.5))
        )
        conditions.append(
            {
                "name": f"p{percent:g}_ocr",
                "scope": "all",
                "percent": percent,
                "kind": "ocr",
                "trial": None,
                "heads": ranked[:n_ablate],
            }
        )

        for scope, pool in random_pools.items():
            if n_ablate > len(pool):
                print(
                    f"skip {scope} random at {percent:g}%: "
                    f"needs {n_ablate} heads but scope has {len(pool)}"
                )
                continue
            trials_here = (
                min(random_trials, 1)
                if n_ablate == len(pool)
                else random_trials
            )
            for trial in range(trials_here):
                rng = random.Random(f"{seed}:{scope}:{percent}:{trial}")
                conditions.append(
                    {
                        "name": f"{scope}_p{percent:g}_random_{trial}",
                        "scope": scope,
                        "percent": percent,
                        "kind": "random",
                        "trial": trial,
                        "heads": rng.sample(pool, n_ablate),
                    }
                )
    return conditions


def get_logits(model, processor, image, heads, head_means, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + PREFILL
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with model.trace(**inputs):
        # nnsight interventions must be registered in model execution order.
        for layer, head in sorted(heads):
            head_dim = model.model.language_model.config.head_dim
            start = head * head_dim
            stop = (head + 1) * head_dim
            model.model.language_model.layers[layer].self_attn.o_proj.input[
                :, :, start:stop
            ] = head_means[layer, start:stop]
        logits = model.output.logits[:, -1].save()
    return logits


def evaluate(model, processor, words, images, condition, head_means, device):
    rows = []
    for word, image in zip(words, images, strict=True):
        logits = get_logits(
            model, processor, image, condition["heads"], head_means, device
        )
        predicted = model.tokenizer.decode(logits[0].argmax(dim=-1))
        correct = word.strip().lower() == predicted.strip().lower()
        rows.append(
            {
                "condition": condition["name"],
                "word": word.strip(),
                "predicted": predicted.strip(),
                "correct": correct,
            }
        )
    return rows


def main():
    args = parse_args()
    if args.n_examples <= 0 or args.random_trials < 0:
        raise ValueError(
            "--n-examples must be positive and --random-trials nonnegative"
        )
    percentages = sorted(set(args.percentages))
    if not percentages or min(percentages) <= 0 or max(percentages) > 100:
        raise ValueError("--percentages must be in (0, 100]")

    name = model_name(args.model)
    score_path = resolve_path(
        args.score_path
        or Path(f"results/score_ocr_heads/{name}/final_1024.pt")
    )
    mean_path = resolve_path(args.head_means or HEAD_MEAN_PATHS[name])
    output_dir = resolve_path(args.output_dir) / name

    scores = load_head_scores(score_path)
    scopes = list(dict.fromkeys(args.scopes))
    all_conditions = build_conditions(
        scores,
        percentages,
        scopes,
        args.random_trials,
        args.random_seed,
    )
    if args.conditions == "ocr":
        conditions = [
            condition
            for condition in all_conditions
            if condition["kind"] in {"baseline", "ocr"}
        ]
    elif args.conditions == "random":
        conditions = [
            condition
            for condition in all_conditions
            if condition["kind"] == "random"
        ]
    else:
        conditions = all_conditions
    print(
        f"scores: {score_path} "
        f"({scores.shape[0]} layers x {scores.shape[1]} heads)"
    )
    for condition in conditions:
        print(f"{condition['name']:<28} {len(condition['heads']):>4} heads")
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "format_version": 1,
        "model": args.model,
        "score_path": str(score_path.resolve()),
        "head_means": str(mean_path.resolve()),
        "percentages": percentages,
        "random_scopes": scopes,
        "random_trials": args.random_trials,
        "n_examples": args.n_examples,
        "stimulus_seed": args.seed,
        "random_seed": args.random_seed,
        "word_list": str(resolve_path(args.word_list).resolve()),
        "backgrounds": (
            str(resolve_path(args.background_dir).resolve())
            if args.background_dir is not None
            else "timm/mini-imagenet:train"
        ),
        "prompt": PROMPT,
        "prefill": PREFILL,
        "ablation_position": "all",
    }
    validate_run_config(output_dir / "run_config.json", run_config)

    import numpy as np
    import pandas as pd
    from datasets import load_dataset
    from nnsight import VisionLanguageModel
    from transformers import AutoProcessor

    sys.path.insert(0, str(ROOT / "scripts"))
    from ocr import filter_raw_words, make_imagenet_image

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_kwargs = {"device_map": args.device, "cache_dir": "/share/NFS/models"}
    if args.device == "cuda":
        model_kwargs["dispatch"] = True
    model = VisionLanguageModel(args.model, **model_kwargs)
    processor = model.processor
    head_means = load_head_means(mean_path, args.device)
    random.seed(args.seed)

    config = model.model.language_model.config
    expected_score_shape = (config.num_hidden_layers, config.num_attention_heads)
    expected_mean_shape = (
        config.num_hidden_layers,
        config.num_attention_heads * config.head_dim,
    )
    if tuple(scores.shape) != expected_score_shape:
        raise ValueError(f"Score shape {tuple(scores.shape)} != {expected_score_shape}")
    if tuple(head_means.shape) != expected_mean_shape:
        raise ValueError(
            f"Mean shape {tuple(head_means.shape)} != {expected_mean_shape}"
        )

    with resolve_path(args.word_list).open(errors="ignore") as file:
        raw_words = [
            word.strip()
            for word in file
            if 2 <= len(word.strip()) <= 8 and "'" not in word
        ]
    words = filter_raw_words(raw_words, model.tokenizer)
    random.shuffle(words)
    words = words[: args.n_examples]
    if len(words) < args.n_examples:
        raise ValueError(f"Only found {len(words)} eligible single-token words")
    image_random_state = random.getstate()

    dataset_kwargs = {"split": "train", "cache_dir": "/share/NFS/datasets"}
    if args.background_dir is None:
        dataset = load_dataset("timm/mini-imagenet", **dataset_kwargs)
    else:
        dataset = ImageDirectory(resolve_path(args.background_dir))
    random.setstate(image_random_state)
    images = [make_imagenet_image(word.strip(), dataset) for word in words]

    results_path = output_dir / "results.csv"
    summaries = {}
    if results_path.exists():
        previous = pd.read_csv(results_path).to_dict("records")
        valid_names = {condition["name"] for condition in all_conditions}
        summaries = {
            row["condition"]: row
            for row in previous
            if row["condition"] in valid_names
        }
    if args.skip_existing:
        pending = [
            condition
            for condition in conditions
            if condition["name"] not in summaries
        ]
        print(f"skipping {len(conditions) - len(pending)} completed conditions")
        conditions = pending
    for index, condition in enumerate(conditions, start=1):
        print(f"[{index}/{len(conditions)}] {condition['name']}")
        rows = evaluate(
            model, processor, words, images, condition, head_means, args.device
        )
        pd.DataFrame(rows).to_csv(
            output_dir / f"{condition['name']}.csv", index=False
        )
        accuracy = sum(row["correct"] for row in rows) / len(rows)
        summaries[condition["name"]] = {
            "model": args.model,
            "condition": condition["name"],
            "scope": condition["scope"],
            "percent": condition["percent"],
            "kind": condition["kind"],
            "trial": condition["trial"],
            "n_heads_ablated": len(condition["heads"]),
            "heads": str(condition["heads"]),
            "n_examples": len(rows),
            "accuracy": accuracy,
        }
        ordered_rows = [
            summaries[condition["name"]]
            for condition in all_conditions
            if condition["name"] in summaries
        ]
        pd.DataFrame(ordered_rows).to_csv(results_path, index=False)
        print(f"accuracy: {accuracy:.3f}")

    print(f"saved results to {output_dir}")


if __name__ == "__main__":
    main()