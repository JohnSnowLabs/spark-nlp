---
layout: model
title: DeBERTa base model
author: John Snow Labs
name: deberta_v3_base_quantized
date: 2023-06-28
tags: [en, english, open_source, embeddings, deberta, v3, base, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.0.0
spark_version: 3.0
supported: true
engine: onnx
annotator: DeBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The DeBERTa model was proposed in [[https://arxiv.org/abs/2006.03654 DeBERTa: Decoding-enhanced BERT with Disentangled Attention]] by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen It is based on Google’s BERT model released in 2018 and Facebook’s RoBERTa model released in 2019. Compared to RoBERTa-Large, a DeBERTa model trained on half of the training data performs consistently better on a wide range of NLP tasks, achieving improvements on MNLI by +0.9% (90.2% vs. 91.1%), on SQuAD v2.0 by +2.3% (88.4% vs. 90.7%) and RACE by +3.6% (83.2% vs. 86.8%).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v3_base_quantized_en_5.0.0_3.0_1687958846162.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_v3_base_quantized_en_5.0.0_3.0_1687958846162.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = DeBertaEmbeddings.pretrained("deberta_v3_base", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
```
```scala
val embeddings = DeBertaEmbeddings.pretrained("deberta_v3_base", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.deberta_v3_base").predict("""Put your text here.""")
```

</div>

{:.model-param}

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = DeBertaEmbeddings.pretrained("deberta_v3_base", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
```
```scala
val embeddings = DeBertaEmbeddings.pretrained("deberta_v3_base", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
```

{:.nlu-block}
```python
import nlu
nlu.load("en.embed.deberta_v3_base").predict("""Put your text here.""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_v3_base_quantized|
|Compatibility:|Spark NLP 5.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|310.7 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## Benchmarking

```bash
Benchmarking
```