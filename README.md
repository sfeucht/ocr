# Using OCR Heads to Verbalize Image Semantics
*Code for https://ocr.baulab.info/*

In this work, we identify attention heads in vision-language models (VLMs) that are responsible for OCR, which is the task of "reading" text in images. But these heads turn out to be useful for visualizing semantics of image tokens in general!

![](https://ocr.baulab.info/images/figure1.png)

# Table of Contents
## Section 2: Finding OCR Heads 
- `sec2__ocr.py` generates images used in this paper for finding OCR heads (random words on ImageNet images), and evalutes models on a small sample. 
- `sec2__score_heads.py` scores attention heads using these randomly-generated images. We have committed scores already in `results/score_ocr_heads`, so you do not have to run this script. 
- `sec2__cache_activations.py` caches averaged head activations across 1000 of these random images for easy mean-ablation. We have also committed the output of this script in `cache/`. 
- `sec2__ablate_pct.py` replicates results from Figure 3.

## Section 3: Verbalization Lens 
- `sec3__lens.ipynb` is a notebook that gives an interactive widget for testing our approach on any image. 
- `sec3__localize_object.ipynb` gives code for visualizing the probability of a particular token across a given image. 
- `sec3__object_detection.py` replicates results from Figure 6. 

## Section 4: Editing Images 
- `sec4__edit_objects.ipynb` provides an example where you can specify an image, a concept to remove, and a concept to add. 
- `sec4__edit_objects_heatmap.py` runs this sweeping across ranks and scaling factors to replicate Figure 8b. 

# Other Info

```
@article{
    feucht2026ocr,
    title={Using OCR Heads to Verbalize Image Semantics},
    author={Sheridan Feucht and Benno Krojer and Sarah Wang and Henry Abrahamsen and Byron C. Wallace and David Bau},
    journal={arXiv preprint},
    year={2026},
    url={https://arxiv.org/abs/2609.18823}
}
```

We focus on Qwen3 models in this repo; if you are having issues replicating our results for other models, or if you have any other questions, please reach out to `feucht.s@northeastern.edu`. 