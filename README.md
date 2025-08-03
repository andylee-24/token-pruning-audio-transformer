# Token Pruning in Audio Transformers: Optimizing Performance and Decoding Patch Importance

by TaeHan Lee and Hyukjun Lee

### Abstract
Vision Transformers (ViTs) have achieved state-of-the-art performance across various computer vision tasks, but their high computational cost remains a challenge. Token pruning has been proposed to reduce this cost by selectively removing less important tokens. While effective in vision tasks by discarding non-object regions, applying this technique to audio tasks presents unique challenges, as distinguishing relevant from irrelevant regions in time-frequency representations is less straightforward. In this study, for the first time, we applied token pruning to ViT-based audio classification models using Mel-spectrograms and analyzed the trade-offs between model performance and computational cost: TopK token pruning can reduce MAC operations of AudioMAE and AST by 30-40%, with less than a 1% drop in accuracy. Our analysis reveals that while high-intensity or high-variation tokens contribute significantly to model accuracy, low-intensity or low variation tokens also remain important when token pruning is applied; pruning solely based on the intensity or variation of signals in a patch leads to a noticeable drop in accuracy. We support our claim by measuring high correlation between attention scores and these statistical features and by showing retained tokens consistently receive distinct attention compared to pruned ones. We also show that AudioMAE retains more low-intensity tokens than AST. This can be explained by AudioMAE's self-supervised reconstruction objective, which encourages attention to all patches, whereas AST's supervised training focuses on label-relevant tokens.

### News
* [03 August 2025] Our paper got accepted for [ECAI-2025](https://ecai2025.org/)! We also uploaded our [checkpoints](https://drive.google.com/drive/folders/1cBDXh98m2qDlYLLX3q6xB-gtU1uUtxhK).
* [02 April  2025] We have released the [preprint](https://arxiv.org/abs/2504.01690) on arXiv.

### Prepare data and pretrained model
* Download pretrained models (ViT-B) from [AudioMAE](https://github.com/facebookresearch/AudioMAE) and [ast](https://github.com/YuanGongND/ast) with stride=16.
* For SPC-2 and ESC-50, preapre datasets following from [AST](https://github.com/YuanGongND/ast).
* For AudioSet and VoxCeleb-1, due to copyright restrictions, we cannot release the data. However, the AudioSet data annotation JSON used in this work is available under   ``audiomae/dataset/audioset``

### Training
* We used miniconda as a python virtual enviornment manager for our experiments.
* Each model requires a different Python environment for training. Please refer to the README.md file under each directory.
* We use a RAM filesystem to preserve disk lifespan. See ``ramdisk.sh`` for details.
* Set the path to your dataset and pretrained model in the training script.

### Extracting features from the model
* Extract Mel-spectrograms, attention scores, and top-k indices using the training scripts.
* Use ``./run_extract_stats.sh`` under audiomae/ to extract visualizations of token prunings, Mel-spectrogram statistics & kendall correlations. Please refer to ``extract_stats.py`` for details

### Citation
```
@article{lee2025tokenpruningaudiotransformers,
      title={Token Pruning in Audio Transformers: Optimizing Performance and Decoding Patch Importance}, 
      author={Taehan Lee and Hyukjun Lee},
      year={2025},
      eprint={2504.01690},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2504.01690}, 
}
```
### Acknowledgement
* Our implementation is based on [AudioMAE](https://github.com/facebookresearch/AudioMAE), [AST](https://github.com/YuanGongND/ast), [EViT](https://github.com/youweiliang/evit) and [DynamicViT](https://github.com/raoyongming/DynamicViT).
