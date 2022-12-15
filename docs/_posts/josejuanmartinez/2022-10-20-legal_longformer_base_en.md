---
layout: model
title: Legal Longformer (Base, 4096)
author: John Snow Labs
name: legal_longformer_base
date: 2022-10-20
tags: [en, open_source]
task: Embeddings
language: en
edition: Spark NLP 4.2.1
spark_version: [3.2, 3.0]
supported: true
annotator: LongformerEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

[Longformer](https://arxiv.org/abs/2004.05150) is a transformer model for long documents. 

`legal_longformer_base` is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents. It supports sequences of length up to 4,096 and it's specifically trained on *legal documents*

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

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legal_longformer_base_en_4.2.1_3.2_1666282710556.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = LongformerEmbeddings\
  .pretrained("legal_longformer_base", "en")\
  .setInputCols(["document", "token"])\
  .setOutputCol("embeddings")\
  .setCaseSensitive(True)\
  .setMaxSentenceLength(4096)
```
```scala
val embeddings = LongformerEmbeddings.pretrained("legal_longformer_base", "en")
  .setInputCols("document", "token") 
  .setOutputCol("embeddings")
  .setCaseSensitive(true)
  .setMaxSentenceLength(4096)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legal_longformer_base|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|531.1 MB|
|Case sensitive:|true|
|Max sentence length:|4096|

## References

https://huggingface.co/saibo/legal-longformer-base-4096