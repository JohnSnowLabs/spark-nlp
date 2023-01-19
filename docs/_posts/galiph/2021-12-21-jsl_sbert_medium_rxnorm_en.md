---
layout: model
title: Sentence Embeddings - sbert medium (tuned)
author: John Snow Labs
name: jsl_sbert_medium_rxnorm
date: 2021-12-21
tags: [embeddings, clinical, licensed, en]
task: Embeddings
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: BertSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps sentences & documents to a 512-dimensional dense vector space by using average pooling on top of BERT model. It's also fine-tuned on the RxNorm dataset to help generalization over medication-related datasets.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_sbert_medium_rxnorm_en_3.3.4_2.4_1640118356633.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/jsl_sbert_medium_rxnorm_en_3.3.4_2.4_1640118356633.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sentence_embeddings = BertSentenceEmbeddings.pretrained("jsl_sbert_medium_rxnorm", "en", "clinical/models")\
        .setInputCols(["sentence"])\
        .setOutputCol("sbert_embeddings")
```
```scala
val sentence_embeddings = BertSentenceEmbeddings.pretrained('jsl_sbert_medium_rxnorm', 'en','clinical/models')\
      .setInputCols("sentence")\
      .setOutputCol("sbert_embeddings")
```
</div>

## Results

```bash
Gives a 512-dimensional vector representation of the sentence.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|jsl_sbert_medium_rxnorm|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|153.9 MB|
|Case sensitive:|false|

## Data Source

Tuned on RxNorm dataset.