---
layout: model
title: DeBERTa large model
author: John Snow Labs
name: deberta_v3_large
date: 2022-03-10
tags: [en, english, deberta, large, v3, embeddings, open_source]
task: Embeddings
language: en
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v3_large_en_3.4.2_3.0_1646903533580.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_v3_large_en_3.4.2_3.0_1646903533580.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = DeBertaEmbeddings.pretrained("deberta_v3_large", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
```
```scala
val embeddings = DeBertaEmbeddings.pretrained("deberta_v3_large", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.deberta_v3_large").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_v3_large|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|1.0 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

[https://huggingface.co/microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)

## Benchmarking

```bash
#### Fine-tuning on NLU tasks

dev results on SQuAD 2.0 and MNLI tasks.

| Model             |Vocabulary(K)|Backbone #Params(M)| SQuAD 2.0(F1/EM) | MNLI-m/mm(ACC)|
|-------------------|----------|-------------------|-----------|----------|
| RoBERTa-large     |50     |304                | 89.4/86.5 | 90.2   |
| XLNet-large       |32     |-                  | 90.6/87.9 | 90.8   |
| DeBERTa-large     |50     |-                  | 90.7/88.0 | 91.3   |
| **DeBERTa-v3-large**|128|304                  |  **91.5/89.0**| **91.8/91.9**|

```