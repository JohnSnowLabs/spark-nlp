---
layout: model
title: DeBERTa base model
author: John Snow Labs
name: deberta_v3_base
date: 2022-03-10
tags: [en, english, open_source, embeddings, deberta, v3, base]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v3_base_en_3.4.2_3.0_1646895494674.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_v3_base_en_3.4.2_3.0_1646895494674.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_v3_base|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|436.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

[https://huggingface.co/microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)

## Benchmarking

```bash
#### Fine-tuning on NLU tasks

dev results on SQuAD 2.0 and MNLI tasks.

| Model             |Vocabulary(K)|Backbone #Params(M)| SQuAD 2.0(F1/EM) | MNLI-m/mm(ACC)|
|-------------------|----------|-------------------|-----------|----------|
| RoBERTa-base      |50     |86                 | 83.7/80.5 | 87.6/-   |
| XLNet-base        |32     |92                 | -/80.2    | 86.8/-   |
| ELECTRA-base      |30    |86                  | -/80.5    | 88.8/    |
| DeBERTa-base      |50     |100                |  86.2/83.1| 88.8/88.5|
| DeBERTa-v3-base   |128|86                       | **88.4/85.4** | **90.6/90.7**|
| DeBERTa-v3-base + SiFT |128|86                | -/- | 91.0/-|

```