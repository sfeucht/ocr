"""
NOTE: the original code used a local copy of the COCO 2014 validation set, but this code uses the 2017 set.
(detection-datasets/coco on HuggingFace). Pass --local to use the local val2014 copy instead.

Using logit lens, measure P(object) in an image across all tokens for a specific layer. If the lens 
is good, then P(object) should be high for images that contain that object, and low for images that do not. 

What we do: 
- for each of the 80 objects, sample n images with the object and n images without the object 
- apply lens to each image token at a particular layer and grab P(object)
- take the max P(object) in this layer 
- show histogram with two colors, images w/ object and images without. 
"""
import json 
import os 
import random 
import torch 
import argparse 
from tqdm import tqdm 
from collections import defaultdict
from PIL import Image 
from datasets import load_dataset
from nnsight import VisionLanguageModel
from transformers import AutoProcessor 

COCO_IMAGES = "/share/NFS/datasets/mscoco/val2014/"
COCO_ANNOT = "/share/NFS/datasets/mscoco/annotations/instances_val2014.json"
IMG_TO_OBJECTS = "/share/NFS/datasets/mscoco/annotations/img_to_objects_val2014.json"

# NOTE: need to go thru dataset and make hashtable of each image id, and all the objects in that image. 
# image_to_objects = defaultdict(list)
# for image_info in coco_info['images']:
#     img_id = image_info['id']
#     for annot in coco_info['annotations']:
#         if img_id == annot['image_id']:
#             image_to_objects[img_id].append(annot['category_id'])

HF_COCO = "detection-datasets/coco"

def load_coco_id(img_id):
    return Image.open(os.path.join(COCO_IMAGES, f"COCO_val2014_{img_id:012d}.jpg"))

# loads val2017 from HF, and builds the same coco_info / img_to_objects structures as the local annotations.
# NOTE: category ids here are HF's contiguous 0-79 labels, not the original COCO ids (1-90 with gaps)
def load_hf_coco():
    # only download the val shards; no_checks because the repo metadata expects a train split too
    ds = load_dataset(HF_COCO, data_files={"val": "data/val-*.parquet"}, split="val", verification_mode="no_checks")
    names = ds.features["objects"]["category"].feature.names

    coco_info = {
        "categories": [{"id": i, "name": name} for i, name in enumerate(names)],
        "images": [],
        "annotations": [],
    }
    img_to_objects = {}
    id_to_row = {}
    meta = ds.select_columns(["image_id", "objects"])
    for row_idx, row in enumerate(meta):
        img_id = row["image_id"]
        id_to_row[img_id] = row_idx
        coco_info["images"].append({"id": img_id})
        img_to_objects[img_id] = row["objects"]["category"]
        for cat in row["objects"]["category"]:
            coco_info["annotations"].append({"image_id": img_id, "category_id": cat})

    def load_image(img_id):
        return ds[id_to_row[img_id]]["image"]

    return coco_info, img_to_objects, load_image

def build_ocr_lens(model, ranking, k, seed=216):
    n_layers = model.model.language_model.config.num_hidden_layers
    n_heads = model.model.language_model.config.num_attention_heads

    if ranking == "all":
        # every head in the model, k is ignored
        idxs = torch.arange(n_layers * n_heads)
    elif ranking in ("random", "random_last50"):
        # sample k random heads, either from anywhere in the model or from the last half of layers
        first_layer = n_layers // 2 if ranking == "random_last50" else 0
        candidates = list(range(first_layer * n_heads, n_layers * n_heads))
        assert k <= len(candidates), f"can't sample {k} heads from {len(candidates)}"
        idxs = torch.tensor(random.Random(seed).sample(candidates, k))
    else:
        # get scores for each head and get top-k
        sum_score, denom = torch.load(ranking)
        head_scores = sum_score / denom 
        _, idxs = torch.topk(head_scores.view(-1), k)

    def flat_to_grid(flatidx, n_heads=n_heads):
        return ((flatidx // n_heads).item(), (flatidx % n_heads).item()) # layer, head 

    hd = model.model.language_model.config.head_dim
    md = model.model.language_model.config.hidden_size
    nrepeats = model.config.text_config.num_attention_heads // model.config.text_config.num_key_value_heads 

    ocr_heads = [flat_to_grid(idx) for idx in idxs]
    ocr_lens = torch.zeros((md, md), device="cuda", dtype=model.dtype)
    for (l, h) in ocr_heads:
        # (out, in) so select columns of the o_proj matrix 
        O = model.model.language_model.layers[l].self_attn.o_proj.weight[:, h * hd : (h+1) * hd]

        # (out, in) GQA with query=16, kv=8, so model dim becomes halved; two q for every kv
        # so (1024, 2048) we have to reshape to be (8, 128, 2048) and repeat_interleave along dim=0
        Vfull = torch.repeat_interleave(
            model.model.language_model.layers[l].self_attn.v_proj.weight.view(-1, hd, md), 
            repeats=nrepeats, dim=0
        ) # (16, 128, 2048)
        assert torch.allclose(Vfull[0], Vfull[1])
        V = Vfull[h]

        ocr_lens += O @ V
    return ocr_lens 


def ocr_ll(model, state, ocr_lens=None):
    if ocr_lens is not None:
        # assert ocr_k > 0 we actually just use torch.eye if ocr_k=0
        state = torch.einsum("od,bsd->bso", ocr_lens, state) 
    return model.lm_head(model.model.language_model.norm(state)).softmax(dim=-1)

# loads in images from COCO if given ids, otherwise takes list of images
def prep_inputs(images, prompt, prefill, processor):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]

    # if given ids, load in the actual Image objects 
    if type(images[0]) == int:
        images = [load_coco_id(img_id) for img_id in images]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text + prefill 

    # nest images per-text sample so both Qwen and Gemma3 processors batch correctly
    inputs = processor(
        text=[text] * len(images), 
        images=[[img] for img in images], 
        return_tensors="pt",
        padding=True
    ).to("cuda")
    
    return inputs 

# for a specific token get max P(token) across a batch of positions at specific layer 
def max_token_prob(images, tok, layer, model, processor, ocr_k=0, ocr_lens=None):
    prompt = ""
    prefill = ""
    inputs = prep_inputs(images, prompt, prefill, processor)

    # just get img pad tokens 
    img_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    mask = inputs["input_ids"] == img_pad_id # each img is different n pad tokens 

    with torch.no_grad():
        with model.trace(**inputs):
            layer_output = model.model.language_model.layers[layer].output.save()

    # get best possible prob of tok across image positions 
    # if ocr_k is 0, equivalent to raw ll 
    probs = ocr_ll(model, layer_output, ocr_lens=ocr_lens)[:, :, tok]

    # before you take max over all the image tokens, exclude non image tokens first. 
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    probs = torch.max(probs, dim=-1).values
    assert len(probs.shape) == 1 
    return probs.tolist()

def get_objects_in_image(img_id, coco_info):
    cate_dict = {}
    for x in coco_info['categories']:
        cate_dict[x['id']] = x['name']

    out = []
    for annot in coco_info['annotations']:
        if annot['image_id'] == img_id:
            out.append(cate_dict[annot['category_id']])
    return out 

def sample_images_with_object(category_id, coco_info, n):
    out_images = []
    for annot_info in coco_info['annotations']:
        if annot_info['category_id'] == category_id:
            out_images.append(annot_info['image_id'])
        if len(out_images) >= n:
            break
    return out_images 

def sample_images_without_object(category_id, coco_info, n, image_to_objects):
    out_images = []
    for image_info in coco_info['images']:
        img_id = image_info['id']
        try:
            if category_id not in image_to_objects[img_id]:
                out_images.append(img_id)
        except KeyError:
            continue # some images don't have object annots I think

        if len(out_images) >= n:
            break 
    return out_images

def main(args):
    random.seed(8)

    # set up model and lens 
    model = VisionLanguageModel(args.model, device_map="cuda", dispatch=True, cache_dir="/share/NFS/models")
    processor = AutoProcessor.from_pretrained(args.model)

    model_name = args.model.split('/')[-1]
    result_dir =f"results/object_detection/{model_name}/"
    if "gaze" in args.ranking:
        result_dir += "gaze/"
    elif args.ranking in ("random", "random_last50"):
        result_dir += f"{args.ranking}/seed-{args.seed}/"
    elif args.ranking == "all":
        result_dir += "all/"
    result_dir += f"ocrk-{args.ocr_k}_layer-{args.layer}_n{args.n}"
    result_dir += "/" if args.local else "_val2017/"
    os.makedirs(result_dir, exist_ok=True)

    metadata = dict(vars(args))
    with open(os.path.join(result_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    ocr_lens = None 
    if args.ocr_k > 0 or args.ranking == "all":
        with torch.no_grad():
            ocr_lens = build_ocr_lens(
                model, args.ranking.format(model_name=model_name), k=args.ocr_k, seed=args.seed
            )

    # set up coco dataset 
    if args.local:
        with open(COCO_ANNOT, "r") as f: 
            coco_info = json.load(f)

        # load in pre-calculated image-to-object dataset info 
        with open(IMG_TO_OBJECTS, "r") as f:
            img_to_objects = json.load(f)
        img_to_objects = {int(k) : v for k, v in img_to_objects.items()}
        load_image = load_coco_id
    else:
        coco_info, img_to_objects, load_image = load_hf_coco()

    # get the token IDs for each of the 80 possible COCO objects, but skip if not single-token. 
    category_tokens = {}
    for x in coco_info['categories']:
        toks = model.tokenizer(' ' + x['name'], add_special_tokens=False)['input_ids']
        if len(toks) == 1:
            category_tokens[x['id']] = toks[0]
    print(len(set(category_tokens.values())), "unique single-token categories")

    with_per_category = defaultdict(list)
    without_per_category = defaultdict(list)

    for category_id, category_tok in category_tokens.items():
        print(category_id, model.tokenizer.decode(category_tok))
        with_imgs = sample_images_with_object(category_id, coco_info, args.n)
        without_imgs = sample_images_without_object(category_id, coco_info, args.n, img_to_objects)
        without_imgs = without_imgs[:len(with_imgs)] # keep balanced; some val2017 categories have < n images

        # all in one go 
        if args.bsz >= len(with_imgs):
            with_per_category[category_id] = max_token_prob(
                [load_image(i) for i in with_imgs], category_tok, args.layer, model, processor, ocr_k=args.ocr_k, ocr_lens=ocr_lens
            )
            without_per_category[category_id] = max_token_prob(
                [load_image(i) for i in without_imgs], category_tok, args.layer, model, processor, ocr_k=args.ocr_k, ocr_lens=ocr_lens
            )

        else: # batched 
            n_batches = len(with_imgs) // args.bsz 
            for batch_idx in tqdm(range(n_batches)):
                with_batch = with_imgs[batch_idx * args.bsz : (batch_idx + 1) * args.bsz]
                without_batch = without_imgs[batch_idx * args.bsz : (batch_idx + 1) * args.bsz]

                with_per_category[category_id] += max_token_prob(
                    [load_image(i) for i in with_batch], category_tok, args.layer, model, processor, ocr_k=args.ocr_k, ocr_lens=ocr_lens
                )
                without_per_category[category_id] += max_token_prob(
                    [load_image(i) for i in without_batch], category_tok, args.layer, model, processor, ocr_k=args.ocr_k, ocr_lens=ocr_lens
                )

        result_dict = {
            "category_tok" : category_tok,
            "category_name" : model.tokenizer.decode(category_tok),
            "with" : with_per_category[category_id],
            "without" : without_per_category[category_id]
        }
        with open(os.path.join(result_dir, f"{category_id}_{model.tokenizer.decode(category_tok).strip()}.json"), "w") as f:
            json.dump(result_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=216)  
    parser.add_argument("--bsz", type=int, default=16)  
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--ocr_k", type=int, default=0) # default: raw ll
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--local", action="store_true") # use local val2014 copy instead of HF val2017
    parser.add_argument("--ranking", type=str, default="results/score_ocr_heads/{model_name}/final_1024.pt") 
    parser.add_argument("--model", choices=[
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
    ], default="Qwen/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()
    main(args)
