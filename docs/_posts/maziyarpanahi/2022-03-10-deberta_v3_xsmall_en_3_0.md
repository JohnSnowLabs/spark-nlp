---
layout: model
title: DeBERTa xsmall model
author: John Snow Labs
name: deberta_v3_xsmall
date: 2022-03-10
tags: [en, english, embeddings, deberta, xsmall, v3, open_source]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v3_xsmall_en_3.4.2_3.0_1646908120895.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = DeBertaEmbeddings.pretrained("deberta_v3_xsmall", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

```
```scala
val embeddings = DeBertaEmbeddings.pretrained("deberta_v3_xsmall", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.deberta_v3_xsmall").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_v3_xsmall|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|169.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

[https://huggingface.co/microsoft/deberta-v3-xsmall](https://huggingface.co/microsoft/deberta-v3-xsmall)

## Benchmarking

```bash
#### Fine-tuning on NLU tasks

The dev results on SQuAD 2.0 and MNLI tasks.

| Model             |Vocabulary(K)|Backbone #Params(M)| SQuAD 2.0(F1/EM) | MNLI-m/mm(ACC)|
|-------------------|----------|-------------------|-----------|----------|
| RoBERTa-base      |50     |86                 | 83.7/80.5 | 87.6/-   |
| XLNet-base        |32     |92                 | -/80.2    | 86.8/-   |
| ELECTRA-base      |30    |86                  | -/80.5    | 88.8/    |
| DeBERTa-base      |50     |100                |  86.2/83.1| 88.8/88.5|
| DeBERTa-v3-large|128|304                      | 91.5/89.0 | 91.8/91.9|
| DeBERTa-v3-base |128|86                       | 88.4/85.4 | 90.6/90.7|
| DeBERTa-v3-small  |128|44                     | 82.8/80.4 | 88.3/87.7|
| **DeBERTa-v3-xsmall** |128|**22**             | **84.8/82.0** | **88.1/88.3**|
| DeBERTa-v3-xsmall+SiFT|128|22                 | -/-       | 88.4/88.5|




```