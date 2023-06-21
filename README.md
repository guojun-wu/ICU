# ICU: Image Caption Understanding

ICU is a divide-and-conquer approach to tackle the challenges of multilingual vision-and-language tasks.

# Setup
We recommend start with a clean environment:

```bash
conda create -n icu python=3.9
conda activate icu
```
Install required packages:
```bash
cd icu
pip install -r requirements.txt
```
# Data Download
## Text Data
The raw text data can be downloaded from [IGLUE](https://github.com/e-bug/iglue/tree/main/datasets)
## Image Data
The raw images can be downloaded from the original websites:
- [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
- [MaRVL](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/42VZ4P)

For XVNLI, we only need part of the images in Flickr30k, the id of the images can be found in text data files.

# Image Captioning
We recommend using OFA with huggingface or ModelScope.
See their [repository](https://github.com/OFA-Sys/OFA) for more details.
We also provide the caption we got with ModelScope in 'data/XVNLI/caption.csv' and 'data/MaRVL/{lang}/caption.csv'.

# Combination of caption and text
For XVNLI, run:
```bash
python data/XVNLI/dataset.py
```
For MaRVL, it will be done automatically when running the icu script.

# ICU
For XVNLI, run:
```bash
python icu.py --task nli --lang {lang} --shot {shot}
```

For MaRVL, run:
```bash
python icu.py --task nli --lang {lang} --frame {frame}
```

# Visualization
We provide a jupyter notebook to visualize the results in 'result.ipynb'.











