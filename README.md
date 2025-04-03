# Token Pruning in Audio Transformers: Optimizing Performance and Decoding Patch Importance

by TaeHan Lee and Hyukjun Lee

### News
* [02 April 2025] We have released the [preprint](https://arxiv.org/abs/2504.01690) on arXiv.

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

### Acknowledgement
* Our implementation is based on [AudioMAE](https://github.com/facebookresearch/AudioMAE), [AST](https://github.com/YuanGongND/ast) and [EViT](https://github.com/youweiliang/evit).
