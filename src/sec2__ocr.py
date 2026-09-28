""" 
This script sets up the basic OCR task that we use to find verbalization heads in Section 2.
You can evaluate a model on these images, and can also run a version of the model with the top-k heads ablated.
"""
import os 
import torch 
import random
import argparse 
import numpy as np 
import pandas as pd 

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor 
from nnsight import VisionLanguageModel, LanguageModel 
from datasets import load_dataset 

# claude helpers to generate an image 
def random_image(ds, size: int) -> Image.Image:
    sample = ds[random.randint(0, len(ds) - 1)]
    img = sample["image"].convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((size, size), Image.LANCZOS)

def contrasting_color(bg_patch: Image.Image) -> tuple[int, int, int]:
    px = list(bg_patch.get_flattened_data())
    avg_lum = sum(0.299 * r + 0.587 * g + 0.114 * b for r, g, b in px) / len(px)
    for _ in range(20):
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        c_lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        if abs(c_lum - avg_lum) > 90:
            return c
    return (0, 0, 0) if avg_lum > 127 else (255, 255, 255)

def make_imagenet_image(text, ds, size: int = 512) -> Image.Image:
    if len(text) > 8:
        raise Exception("text must be leq 8 chars")

    img = random_image(ds, size)
    draw = ImageDraw.Draw(img)

    # Random font size — conservative range, scales with image size
    font_size = random.randint(int(size * 0.10), int(size * 0.22))
    font = None
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # linux
        "/Users/sfeucht/Library/Fonts/DejaVuSans.ttf",      # mac
        "/System/Library/Fonts/Helvetica.ttc",              # mac fallback
    ):
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default(font_size)  # Pillow >= 10 honors size

    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Random position — keep a small margin so text never clips the edge
    margin = int(size * 0.02)
    max_x = size - w - margin - bbox[0]
    max_y = size - h - margin - bbox[1]
    x = random.randint(margin - bbox[0], max(margin - bbox[0], max_x))
    y = random.randint(margin - bbox[1], max(margin - bbox[1], max_y))

    text_region = img.crop((
        max(0, int(x + bbox[0])), max(0, int(y + bbox[1])),
        min(size, int(x + bbox[0] + w)), min(size, int(y + bbox[1] + h)),
    ))
    color = contrasting_color(text_region)

    draw.text((x, y), text, fill=color, font=font)
    return img

def make_white_image(text: str, size: int = 512, font_size: int = 80) -> Image.Image:
    text = text.strip()
    if len(text) > 8:
        raise Exception("text must be leq 8 chars")

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - w) / 2 - bbox[0], (size - h) / 2 - bbox[1]), text, fill="black", font=font)

    return img

# NOTE: this adds spaces in front, but a lot of 中文 tokens are MTW if you add a space.
# basically, this kind of assumes latin character tokens. 
def filter_raw_words(raw_words, tokenizer, multitok=False, include_title=False, max_len=8, add_space=True):
    final_words = []

    # lowercase version 
    for w in raw_words:
        ww = w.lower()
        if add_space:
            ww = ' ' + ww 
        tokenized = tokenizer(ww, add_special_tokens=False)['input_ids']
        if (multitok or len(tokenized) == 1) and len(w) <= max_len:
            final_words.append(ww)

    # title case version 
    if include_title:
        for w in raw_words:
            ww = w.title()
            if add_space: 
                ww = ' ' + ww 
            tokenized = tokenizer(ww, add_special_tokens=False)['input_ids']
            if (multitok or len(tokenized) == 1) and len(w) <= max_len:
                final_words.append(ww)

    # how many "duplicates?"
    n_duplicates = 0
    for w in final_words:
        if w.lower() != w:
            for w2 in final_words:
                if w.lower() == w2: 
                    n_duplicates += 1 
                    break 
    print(f"{n_duplicates} lower/title duplicates out of {len(final_words)} single token words")
    # print(random.sample(single_tok_words, k=10))
    return final_words 

# for a given prompt, (image), and prefill, get the model's predictions. possibly with ablated heads.
def get_logits(model, prompt, prefill, img=None, processor=None, heads_to_ablate=None, head_means=None):
    if "VL" not in model.config._name_or_path:
        messages = [{
            "role": "user",
            "content": prompt
        }]

        text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + prefill
        inputs = model.tokenizer(text, return_tensors="pt").to("cuda")

    elif "VL" in model.config._name_or_path:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + prefill 

        if img is not None:
            inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda")
        else:
            inputs = processor(text=[text], return_tensors="pt").to("cuda")

    if heads_to_ablate is not None:
        n_layers = model.model.language_model.config.num_hidden_layers
        headdim = model.model.language_model.config.head_dim
        unique_layers = sorted(list(set([l for (l, h) in heads_to_ablate])))

        with model.trace(**inputs):
            for layer in range(n_layers):
                if layer in unique_layers:
                    for (l, h) in heads_to_ablate:
                        if l == layer:
                            # ablate this head at all token positions. 
                            model.model.language_model.layers[l].self_attn.o_proj.input[
                                :, :, h * headdim : (h + 1) * headdim
                            ] = head_means[l, h * headdim : (h + 1) * headdim]
            
            logits = model.output.logits.save() 
    else:
        with model.trace(**inputs):
            logits = model.output.logits.save()

    return logits 

# generate instead of trace 
def generate_pred(model, prompt, prefill, img=None, processor=None):
    if "VL" not in model.config._name_or_path:
        messages = [{
            "role": "user",
            "content": prompt
        }]

        text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + prefill
        inputs = model.tokenizer(text, return_tensors="pt").to("cuda")

    elif "VL" in model.config._name_or_path:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + prefill 

        if img is not None:
            inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda")
        else:
            inputs = processor(text=[text], return_tensors="pt").to("cuda")

    generated = model._model.generate(**inputs, max_new_tokens=64)
    generated_new = generated[0][len(inputs.input_ids[0]):]
    return generated_new 


def main(args):
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)

    min_word_len = 2
    model_name = args.model.split('/')[-1]

    save_dir = f"results/evaluate_ocr/{model_name}/"
    os.makedirs("images/evaluate_ocr/", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    with open("/usr/share/dict/words") as f:
        word_list = [w.strip() for w in f if (w.strip() and "'" not in w)]
    word_list = [w for w in word_list if len(w) >= min_word_len]
    print(f"{len(word_list)} raw words from {min_word_len}-8 chars loaded")


    processor = None 
    if "VL" in args.model:
        print("Loading VLM...")
        model = VisionLanguageModel(args.model, device_map="cuda", dispatch=True)
        processor = AutoProcessor.from_pretrained(model.config._name_or_path)
    else:
        print("Loading vanilla LLM...")
        model = LanguageModel(args.model, device_map="cuda", dispatch=True)

    # load in head means and scores if needed 
    n_heads = model.model.language_model.config.num_attention_heads
    def flat_to_grid(flatidx, n_heads=n_heads):
        return ((flatidx // n_heads).item(), (flatidx % n_heads).item()) # layer, head 
    head_means = None 
    heads_to_ablate = None 

    if args.ablate_k > 0:
        if args.multitok:
            raise Exception("ablating with multitok not implemented.")
        try:
            cache_dir = {
                "Qwen3-VL-2B-Instruct" : "cache/Qwen3-VL-2B-Instruct/images__find_ocr_heads__Qwen3-VL-2B-Instruct_seed177/examples2100.pt",
                "Qwen3-VL-8B-Instruct" : "cache/Qwen3-VL-8B-Instruct/images__find_ocr_heads__Qwen3-VL-8B-Instruct_seed177/examples2100.pt"
            }[model_name]
            head_means = torch.load(cache_dir)
        except KeyError:
            raise Exception("please indicate a mean head activation file to ablate using.")

        try:
            if args.ablate_score_path != "":
                head_scores, _ = torch.load(args.ablate_score_path)
                abbrv_scores = '_' + args.ablate_score_path.split('/')[-1]
            else:
                head_scores, _ = torch.load(f"results/find_ocr_heads/{model_name}/examples1500_seed177_final.pt")

            vals, idxs = torch.topk(head_scores.view(-1), args.ablate_k)
            vals = [v.item() for v in vals]
            heads_to_ablate = [flat_to_grid(idx) for idx in idxs]
            print("ablating", list(zip(heads_to_ablate, vals)))
        except FileNotFoundError:
            raise Exception("Could not find scores for this model.")

        save_dir = os.path.join(save_dir, f"ablated{abbrv_scores}/")
        os.makedirs(save_dir, exist_ok=True)

    if args.image == "whitebg":
        assert "VL" in args.model
        print("giving image input to VLM")
    elif args.image == "imagenet":
        ds = load_dataset("timm/mini-imagenet", split="train")

    single_tok_words = filter_raw_words(word_list, model.tokenizer, multitok=args.multitok)
    random.shuffle(single_tok_words)
    
    ctr = 0
    corr = 0 
    to_save = []
    for w in single_tok_words:
        img = None 
        if args.image == "whitebg":
            img = make_white_image(w.strip())
            img.save(f"images/evaluate_ocr/{w.strip()}.png")
        elif args.image == "imagenet":
            img = make_imagenet_image(w.strip(), ds)
            img.save(f"images/evaluate_ocr/{w.strip()}.png")

        prompt = f"Transcribe the text in this image."
        prefill = "Text:"
        ans = w

        if args.multitok:
            generated_toks = generate_pred(model, prompt, prefill, img, processor=processor)
            pred_s = model.tokenizer.decode(generated_toks)

            is_correct = ans.strip().lower() in pred_s.strip().lower()
            if is_correct:
                corr += 1 

        else:
            logits = get_logits(
                model, prompt, prefill, img, processor=processor,
                heads_to_ablate=heads_to_ablate, head_means=head_means
            )
            pred_s = model.tokenizer.decode(logits[0, -1].argmax(dim=-1))

            is_correct = ans.strip().lower() == pred_s.strip().lower()
            if is_correct:
                corr += 1 
        
        print(prompt, repr(ans), repr(pred_s))

        to_save.append([w, prompt, prefill, ans, pred_s, is_correct])
        
        ctr += 1 
        if ctr > args.n_examples:
            break 
    
    print(f'acc: {corr / ctr}')

    df = pd.DataFrame(to_save, columns=["word", "prompt", "prefill", "answer", "predicted", "correct"])

    fname = f"{args.image}_{args.n_examples}"
    fname += "_multitok" if args.multitok else ""
    if args.ablate_k > 0:
        fname += f"_ablate{args.ablate_k}"
    df.to_csv(os.path.join(save_dir, f"{fname}.csv"))

    with open(os.path.join(save_dir, f"{fname}.txt"), 'w') as f:
        f.write(f"acc: {corr / ctr} ({corr}/{ctr})")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
    ], default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--image", choices=["whitebg", "imagenet"], default="imagenet")
    parser.add_argument("--multitok", action="store_true", default=False)
    parser.add_argument("--ablate_k", default=0, type=int) # top-k heads to ablate based on default scores 
    parser.add_argument("--ablate_score_path", default="", type=str) # top-k heads to ablate based on default scores 
    parser.add_argument("--n_examples", type=int, default=100)
    args = parser.parse_args()
    main(args)