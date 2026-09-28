"""CLAUDE SCRIPT
Sweep [rank x scale] for the OCR-lens object substitution edit. Over ImageNet images.

Same intervention as ``src/sec4__edit_objects.ipynb``: we take an imagenet image of e.g. a
motorcycle, remove the " motorcycle" direction from every image token (through
pinv of the rank-truncated OCR lens) and add back " dog" scaled by ``scale``.

We then get the model to generate a description of the image. 
"""
import argparse
import json
import os
import random 

import torch
from tqdm import tqdm
from collections import defaultdict
from nnsight import VisionLanguageModel

from sec3__object_detection import build_ocr_lens, prep_inputs, max_token_prob

HF_CACHE = "/share/NFS/datasets"

DEFAULT_RANK_PCTS = [0.01, 0.05, 0.10, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
DEFAULT_SCALES = [1, 2, 3, 4, 5, 10]
DEFAULT_OCR_K = {
    "Qwen3-VL-2B-Instruct": 45,
    "Qwen3-VL-8B-Instruct": 115,
}

def make_singletok(model):
    """True iff `' ' + s` is a single token for this tokenizer."""
    def singletok(s):
        toks = model.tokenizer(" " + s, add_special_tokens=False)['input_ids']
        return len(toks) == 1
    return singletok

def imagenet_sampler(model, n, rng, repo="ILSVRC/imagenet-1k", split="validation"):
    """Yield (image, image_key, remove_str, add_str) from imagenet-1k.

    imagenet gives exactly one label per image, so remove_str is that image's
    class and add_str is any other single-token class. Class names in the HF
    copy are comma-separated synonym lists ("tench, Tinca tinca"); we keep the
    first synonym. `ILSVRC/imagenet-1k` is gated -- accept the terms on the Hub
    and `huggingface-cli login` first, or pass a non-gated mirror such as
    `evanarlian/imagenet_1k_resized_256`.
    """
    from datasets import load_dataset

    ds = load_dataset(repo, split=split, cache_dir=HF_CACHE)
    singletok = make_singletok(model)

    label_names = [n.split(',')[0].strip() for n in ds.features['label'].names]
    singletok_labels = sorted({n for n in label_names if singletok(n)})
    if len(singletok_labels) < 2:
        raise RuntimeError("not enough single-token imagenet classes for this tokenizer")

    yielded = 0
    while yielded < n:
        idx = rng.randrange(len(ds))
        ex = ds[idx]
        remove_str = label_names[ex['label']]
        if not singletok(remove_str):
            continue

        add_str = rng.choice([l for l in singletok_labels if l != remove_str])
        yield (
            ex['image'].convert("RGB"), idx,
            " " + remove_str, " " + add_str,
        )
        yielded += 1

def load_imagenet_image(imagenet_idx, repo="ILSVRC/imagenet-1k", split="validation"):
    from datasets import load_dataset
    
    ds = load_dataset(repo, split=split, cache_dir=HF_CACHE)
    return ds[imagenet_idx]['image'].convert("RGB")

def substitute_generate(
        model, inp, from_layer, until_layer, scale_new,
        remove_str, add_str, pinv_func, rank
    ):
    img_pad_id = model.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    img_tok_mask = inp["input_ids"] == img_pad_id  # each img is different n pad tokens

    remove_tok = model.tokenizer(remove_str)['input_ids'][0]
    add_tok = model.tokenizer(add_str)['input_ids'][0]

    remove_u = model.lm_head.weight[remove_tok]
    add_u = model.lm_head.weight[add_tok]

    remove_v = pinv_func(rank, remove_u)
    add_v = pinv_func(rank, add_u)

    with torch.no_grad(), model.generate(**inp, max_new_tokens=500, do_sample=False):
        for layer in range(from_layer, until_layer):
            img_states = model.model.language_model.layers[layer].input[img_tok_mask].save()

            # always do ocr_replace_tok, if we want raw ll just do torch.eye 
            scalars_per_tok = (img_states @ remove_v) / (remove_v @ remove_v)
            img_states = img_states - scalars_per_tok[:, None] * remove_v
            img_states = img_states + scalars_per_tok[:, None] * scale_new * add_v

            model.model.language_model.layers[layer].input[img_tok_mask] = img_states

        out = model.generator.output.save()

    prompt_len = inp['input_ids'].shape[-1]
    return model.tokenizer.decode(out[0, prompt_len:])

def run_free_generate(
        image, image_key, remove_str, add_str, model, ocr_lens, ranks, scales,
        from_layer, until_layer, metadata_dict, pinv_func
    ):
    question = "Describe this image."
    inp = prep_inputs([image], question, "", model.processor)

    # out of curiosity, see what max token prob was at layer 0 under our lens
    original_rem_prob = max_token_prob(
        [image], model.tokenizer(remove_str, add_special_tokens=False)['input_ids'][0],
        0, model, model.processor, ocr_k=metadata_dict["ocr_k"], ocr_lens=ocr_lens.bfloat16()
    )[0]

    # save metadata
    metadata_dict["image_id"] = image_key
    metadata_dict["remove_str"] = remove_str 
    metadata_dict["add_str"] = add_str
    metadata_dict["question"] = question 
    metadata_dict["original_rem_prob"] = original_rem_prob

    results = {
        "metadata": metadata_dict,
        "original_rem_prob" : original_rem_prob,
        # dict with rank keys and each list covers all the scales
        "generations" : defaultdict(list)
    }

    # get vanilla pass, just using raw logit lens unembed 
    noop = lambda _, v: v 
    results["vanilla_generation"] = substitute_generate(
        model, inp, from_layer, until_layer, 1, 
        remove_str, add_str, noop, None
    )

    # all the interveneds 
    for rank in ranks:
        for scale in tqdm(scales, desc="scale"):
            generated_str = substitute_generate(
                model, inp, from_layer, until_layer, scale,
                remove_str, add_str, pinv_func, rank
            )

            results["generations"][rank].append(generated_str)

    return results 

def main(args):
    random.seed(args.seed)

    model_name = args.model.split('/')[-1]
    model = VisionLanguageModel(args.model, device_map="cuda", dispatch=True,
                                cache_dir="/share/NFS/models")

    n_layers = model.config.text_config.num_hidden_layers
    model_dim = model.config.text_config.hidden_size
    until_layer = args.until_layer if args.until_layer is not None else n_layers

    ocr_k = args.ocr_k if args.ocr_k is not None else DEFAULT_OCR_K[model_name]
    if ocr_k > 0:
        ocr_lens = build_ocr_lens(model, args.ranking.format(model_name=model_name), ocr_k)
    else:
        ocr_lens = torch.eye(model_dim, device=model.device)

    # claude: you don't actually have to torch.pinv, just directly use the small low-rank matrices 
    # equivalent to doing `v_vec = pinv(trunc_k) @ u_vec`
    U, Sigma, Vh = torch.linalg.svd(ocr_lens.float())
    def pinv_func(k, u_vec):
        return (Vh[:k, :].T @ ((U[:, :k].T @ u_vec.float()) / Sigma[:k])).to(model.dtype)

    # get singular values of the lens and get all the pcts 
    Sigma_sq = torch.square(torch.linalg.svdvals(ocr_lens.float()))
    denom = Sigma_sq.sum(dim=-1)
    ranks = []
    for rankpct in args.rank_pcts:
        if rankpct == 1.0:
            ranks.append(model_dim)
        else:
            for k in range(len(Sigma_sq)):
                if Sigma_sq[:k].sum(dim=-1) / denom >= rankpct:
                    ranks.append(k)
                    break 
    print("ranks", list(zip(ranks, args.rank_pcts)))
    
    scales = args.scales

    result_dir = f"results/edit_objects_heatmap/{model_name}/"
    if args.scales != DEFAULT_SCALES:
        result_dir = os.path.join(result_dir, f"scales_{'-'.join([str(scl) for scl in args.scales][:10])}")
    if args.rank_pcts != DEFAULT_RANK_PCTS:
        result_dir = os.path.join(result_dir, f"rankpcts_{'-'.join([str(scl) for scl in args.rank_pcts][:10])}")
    os.makedirs(result_dir, exist_ok=True)

    metadata_dict = dict(vars(args)) | {
        "model_name": model_name, "ocr_k": ocr_k, "n_layers": n_layers,
        "model_dim": model_dim, "until_layer": until_layer,
        "ranks": ranks, "scales": scales, "dataset": "imagenet",
    }

    rng = random.Random(args.seed)
    samples = imagenet_sampler(model, args.sample_images, rng,
                               repo=args.imagenet_repo, split=args.imagenet_split)

    # if told to replicate a previous run, load in the same dataset, image_id, and add/rm
    # but now we'll do it with whatever settings (i.e. ocr_k=0)
    if len(args.previous_run) > 0:
        assert args.sample_images == 0

        with open(args.previous_run, "r") as f:
            prev_metadata = json.load(f)["metadata"]

        assert prev_metadata["model"] == args.model 
        # old runs without a dataset key were coco, which is no longer supported
        assert prev_metadata.get("dataset") == "imagenet", "previous run must be an imagenet run"
        print(prev_metadata)

        # image_id is an index into the dataset, so load it from the same repo/split as the previous run
        image_idx = prev_metadata["image_id"]
        img = load_imagenet_image(
            image_idx,
            repo=prev_metadata.get("imagenet_repo", args.imagenet_repo),
            split=prev_metadata.get("imagenet_split", args.imagenet_split),
        )
        
        samples = [(img, image_idx, prev_metadata["remove_str"], prev_metadata["add_str"])]

    for image, image_key, remove_str, add_str in samples:
        out_fname = (f"img-imagenet-{image_key}_ocrk-{ocr_k}"
                     f"_rem{remove_str.strip()}_add{add_str.strip()}.json")

        out_path = os.path.join(result_dir, out_fname)
        if not args.overwrite and os.path.exists(out_path):
            print("skipping (already exists)", out_path)
            continue

        results = run_free_generate(
            image, image_key, remove_str, add_str, model, ocr_lens, ranks, scales,
            args.from_layer, until_layer, metadata_dict, pinv_func
        )

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--ranking", default="results/score_ocr_heads/{model_name}/final_1024.pt")
    parser.add_argument("--ocr_k", type=int, default=None,
                        help="number of OCR heads in the lens (default: per-model value)")
    parser.add_argument("--previous_run", type=str, default="")
    parser.add_argument("--sample_images", type=int, default=100,
                        help="number of imagenet images to sample (set to 0 with --previous_run)")
    parser.add_argument("--imagenet_repo", default="ILSVRC/imagenet-1k",
                        help="HF repo for imagenet (gated; non-gated mirror: "
                             "evanarlian/imagenet_1k_resized_256)")
    parser.add_argument("--imagenet_split", default="validation")
    parser.add_argument("--from_layer", type=int, default=0)
    parser.add_argument("--until_layer", type=int, default=None)
    parser.add_argument("--rank_pcts", type=float, nargs="+", default=DEFAULT_RANK_PCTS)
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--seed", type=int, default=216)
    parser.add_argument("--overwrite", action="store_true",
                        help="redo samples whose output json already exists (default: skip them)")
    main(parser.parse_args())
