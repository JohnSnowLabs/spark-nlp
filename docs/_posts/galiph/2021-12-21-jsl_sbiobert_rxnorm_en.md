---
layout: model
title: Sentence Embeddings - sbiobert (tuned)
author: John Snow Labs
name: jsl_sbiobert_rxnorm
date: 2021-12-21
tags: [embeddings, clinical, licensed, en]
task: Embeddings
language: en
edition: Spark NLP for Healthcare 3.3.4
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained to generate contextual sentence embeddings of input sentences.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_sbiobert_rxnorm_en_3.3.4_2.4_1640103567278.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sentence_embeddings = BertSentenceEmbeddings.pretrained("jsl_sbiobert_rxnorm", "en", "clinical/models")\
        .setInputCols(["sentence"])\
        .setOutputCol("sbioert_embeddings")

```

</div>

## Results

```bash
Gives a 768-dimensional vector representation of the sentence.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|jsl_sbiobert_rxnorm|
|Compatibility:|Spark NLP for Healthcare 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.0 MB|

## Data Source

Tuned on RxNorm dataset.