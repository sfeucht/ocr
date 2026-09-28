"""
For all attention heads in a given model, save the mean of that head's activations
across all token positions when it is transcribing from an image. 

In order to run this script, you have to have images in the images/ directory, which means you must have run ocr.py.
If you want to *ablate* attention heads in ocr.py or sec2__ablate_pct.py, you must run this first to allow mean-ablations to work.
"""
import os 
import argparse 
import torch 

from PIL import Image 
from nnsight import VisionLanguageModel
from sec2__score_heads import prep_inputs

# get mean activations for a batch 
def mean_of_batch(model, processor, prompt, prefill, images):
    inputs = prep_inputs(processor, prompt, prefill, images)

    # for 4B, model_dim != head_dim * n_heads :o 
    # you have 32 heads, 128-dim. model.model.language_model.config.hidden_size
    textcfg = model.model.language_model.config 
    n_layers = textcfg.num_hidden_layers
    headacts_dim = textcfg.num_attention_heads * textcfg.head_dim 

    mean_acts = torch.zeros((n_layers, headacts_dim), device="cuda")

    with model.trace(**inputs):
        for layer in range(n_layers):
            head_acts = model.model.language_model.layers[layer].self_attn.o_proj.input
            mean_acts[layer] += head_acts.mean(dim=0).mean(dim=0).save() # [bsz, seq, dim]

    return mean_acts 

def main(args):
    model = VisionLanguageModel(args.model, device_map="cuda")
    processor = model.processor 
    model_name = args.model.split('/')[-1]
    image_dir = os.path.join(args.image_dir, f"{model_name}_seed{args.image_seed}")

    # default from `score_ocr_heads.py`
    prompt = f"Transcribe the word in this image."
    prefill = "Word:"

    all_images = [Image.open(os.path.join(image_dir, i)) for i in os.listdir(image_dir)]

    # for 4B, model_dim != head_dim * n_heads :o 
    # you have 32 heads, 128-dim. model.model.language_model.config.hidden_size
    textcfg = model.model.language_model.config 
    n_layers = textcfg.num_hidden_layers
    headacts_dim = textcfg.num_attention_heads * textcfg.head_dim 

    with torch.no_grad():
        accumulator = torch.zeros((n_layers, headacts_dim), device="cuda") # (n_layers, n_heads, head_dim)
        batches_seen = 0 

        n_batches = len(all_images) // args.bsz # drop last batch 

        for batch_idx in range(n_batches):
            batch_images = all_images[(batch_idx * args.bsz) : (batch_idx * args.bsz) + 1] 

            batch_acts = mean_of_batch(model, processor, prompt, prefill, batch_images)
            accumulator += batch_acts 
            batches_seen += 1 

            if batches_seen * args.bsz > args.max_examples:
                break 
        
        accumulator /= batches_seen 

        model_name = args.model.split('/')[-1]
        image_dir_out = image_dir.replace('/', '__')
        save_dir = f"cache/{model_name}/{image_dir_out}/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(accumulator, save_dir + f"examples{batches_seen * args.bsz}.pt")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
    ], default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--image_dir", type=str, default="images/score_ocr_heads")
    parser.add_argument("--image_seed", type=int, default=177)
    parser.add_argument("--bsz", type=int, default=200)
    parser.add_argument("--max_examples", type=int, default=10000)
    args = parser.parse_args()
    main(args)