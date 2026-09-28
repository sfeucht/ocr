"""
For a given ocr.py type of image, score each head by how much it outputs the word in that image.

Example: an image with the word "her":
_________
|       |
|  her  | --> <user>Transcribe the word in this image. <assistant>Word: 
|_______|

We apply logit lens to the output of each attention head at the last token position, 
calculating P(`her`) for that particular head. 

The "OCR Score" for a given head is just this P(word) score averaged across many images.
"""
import os
import re
import glob
import random
import argparse
import torch 
import numpy as np 
from tqdm import tqdm 

from transformers import AutoProcessor 
from nnsight import VisionLanguageModel 
from datasets import load_dataset 
from ocr import filter_raw_words, make_imagenet_image

def prep_inputs(processor, prompt, prefill, images):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text + prefill 
    inputs = processor(text=[text] * len(images), images=[[img] for img in images], return_tensors="pt").to("cuda")
    return inputs 

def ll(model, state):
    return model.lm_head(model.model.language_model.norm(state)).softmax(dim=-1)

def sum_of_head_probs(model, processor, prompt, prefill, words, images):
    tokens = [model.tokenizer(w, add_special_tokens=False)['input_ids'][0] for w in words]

    bsz = len(images)
    inputs = prep_inputs(processor, prompt, prefill, images)

    # set up accumulator 
    textcfg = model.config.text_config
    headdim = textcfg.head_dim
    nheads = textcfg.num_attention_heads
    all_probs = torch.zeros(
        (textcfg.num_hidden_layers, nheads), device="cuda"
    )

    with torch.no_grad():
        # save all the head outputs at each layer, just at last token position 
        all_head_outputs = []
        with model.trace(**inputs):
            for layer in range(textcfg.num_hidden_layers):
                # (bsz, ~seq~, n_heads*headdim)
                head_output = model.model.language_model.layers[layer].self_attn.o_proj.input[:, -1] 
                hd = headdim  # same head dim for all layers in other models 

                all_head_outputs.append(
                    head_output.view(head_output.shape[0], nheads, hd).save()
                )

        for layer in tqdm(range(textcfg.num_hidden_layers)):
            this_layer_module = model.model.language_model.layers[layer]

            hd = headdim  # same head dim for all layers in other models 

            # get all the O slices for this layer. (out=hidden_size, in=nheads*headdim)
            O = this_layer_module.self_attn.o_proj.weight
            o_slices = O.view(O.shape[0], nheads, hd) # for head h do o_slices[:, h, :]

            # now project to model dimension. each head output (bsz, n_heads[i], headdim) gets
            # multiplied by its respective O columns (hidden_size, nheads[i], headdim)
            projected_output = torch.einsum("bnh,dnh->bnd", all_head_outputs[layer], o_slices)

            # now we have (bsz, n_heads, hidden_size) from last tok position which we can logit lens to get (bsz, n_heads) list of probs 
            desired_probs = ll(model, projected_output)[torch.arange(bsz), :, tokens].save()

            assert (len(desired_probs.shape) == 2) and (desired_probs.shape[0] == bsz) and (desired_probs.shape[1] == nheads)
            all_probs[layer] = desired_probs.sum(dim=0)

        return all_probs


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.model.split('/')[-1]

    img_save_dir = f"images/score_ocr_heads/{model_name}_seed{args.seed}/"
    result_dir =f"results/score_ocr_heads/{model_name}/"
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    min_word_len = 2
    prompt = f"Transcribe the word in this image."
    prefill = "Word:"

    with open("/usr/share/dict/words") as f:
        word_list = [w.strip() for w in f if (w.strip() and "'" not in w)]
    word_list = [w for w in word_list if len(w) >= min_word_len]
    print(f"{len(word_list)} raw words from {min_word_len}-8 chars loaded")

    model = VisionLanguageModel(args.model, device_map="cuda", dispatch=True)
    processor = AutoProcessor.from_pretrained(model.config._name_or_path)

    ds = load_dataset("timm/mini-imagenet", split="train")

    single_tok_words = filter_raw_words(word_list, model.tokenizer, multitok=False, add_space=True)
    single_tok_words = random.sample(single_tok_words, k=args.n_examples)

    bsz = args.bsz 
    if args.bsz > args.n_examples:
        bsz = args.n_examples 

    n_batches = len(single_tok_words) // bsz

    # for saving results
    fname = f"examples{args.n_examples}_seed{args.seed}_bsz{bsz}"

    # we want to accumulate average probability differences for each head
    textcfg = model.config.text_config
    all_probdiffs = torch.zeros((textcfg.num_hidden_layers, textcfg.num_attention_heads), device="cuda")
    n_examples = 0 
    start_batch_idx = 0

    existing_batches = glob.glob(result_dir + fname + "_batch*")
    if len(existing_batches) > 0:
        start_batch_idx = max([int(re.search(r"batch(\d+)\.pt", s).group(1)) for s in existing_batches])
        all_probdiffs, n_examples = torch.load(result_dir + fname + f"_batch{start_batch_idx}.pt")
        all_probdiffs = all_probdiffs.to("cuda")

    for batch_idx in range(n_batches):
        if batch_idx < start_batch_idx:
            continue # skip if already done 

        # for this batch of words, generate images. 
        words = single_tok_words[batch_idx * bsz : (batch_idx + 1) * bsz]
        images = []
        for w in words:
            img = make_imagenet_image(w.strip(), ds)
            img.save(img_save_dir + f"{w.strip()}.png")
            images.append(img)

        with torch.no_grad():
            all_probdiffs += sum_of_head_probs(model, processor, prompt, prefill, words, images)
            n_examples += len(words)
    
        # save intermediate checkpoint
        torch.save((all_probdiffs.cpu(), n_examples), result_dir + fname + f"_batch{batch_idx}.pt")

    # save as tuple with denominator, with easy-to-remember fname as well for future scripts. 
    torch.save((all_probdiffs.cpu(), n_examples), result_dir + fname + "_final.pt")
    torch.save((all_probdiffs.cpu(), n_examples), result_dir + "final_1024.pt")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
    ], default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--n_examples", type=int, default=1024)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--seed", type=int, default=177)
    args = parser.parse_args()
    main(args)