---
layout: model
title: Longformer Base (longformer_base_4096)
author: John Snow Labs
name: longformer_base_4096
date: 2021-08-04
tags: [longformer, base, embeddings, transformer, en, english, open_source]
task: Embeddings
language: en
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: LongformerEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

[Longformer](https://arxiv.org/abs/2004.05150) is a transformer model for long documents. 

`longformer_base_4096` is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents. It supports sequences of length up to 4,096. 

Longformer uses a combination of a sliding window (local) attention and global attention. Global attention is user-configured based on the task to allow the model to learn task-specific representations.

If you use `Longformer` in your research, please cite [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150).
```
@article{Beltagy2020Longformer,
title={Longformer: The Long-Document Transformer},
author={Iz Beltagy and Matthew E. Peters and Arman Cohan},
journal={arXiv:2004.05150},
year={2020},
}
```

`Longformer` is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/longformer_base_4096_en_3.2.0_2.4_1628093002279.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/longformer_base_4096_en_3.2.0_2.4_1628093002279.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = LongformerEmbeddings\
.pretrained("longformer_base_4096")\
.setInputCols(["document", "token"])\
.setOutputCol("embeddings")\
.setCaseSensitive(True)\
.setMaxSentenceLength(4096)
```
```scala
val embeddings = LongformerEmbeddings.pretrained("longformer_base_4096", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")
.setCaseSensitive(true)
.setMaxSentenceLength(4096)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.longformer").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|longformer_base_4096|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|4096|

## Data Source

[https://huggingface.co/allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096)
