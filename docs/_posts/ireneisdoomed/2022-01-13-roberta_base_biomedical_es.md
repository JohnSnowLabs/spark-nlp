---
layout: model
title: RoBERTa base biomedical
author: ireneisdoomed
name: roberta_base_biomedical
date: 2022-01-13
tags: [es, open_source]
task: Text Classification
language: es
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a RoBERTa-based model trained on a biomedical corpus in Spanish.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/ireneisdoomed/roberta_base_biomedical_es_3.4.0_3.0_1642093372752.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
    .setInputCol("term")\
    .setOutputCol("document")

tokenizer = Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")

roberta_embeddings = RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es", "@ireneisdoomed")\
    .setInputCols(["document", "token"])\
    .setOutputCol("roberta_embeddings")

pipeline = Pipeline(stages = [
    documentAssembler,
    tokenizer,
    roberta_embeddings])
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_biomedical|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[document, token]|
|Output Labels:|[embeddings]|
|Language:|es|
|Size:|301.7 MB|

## Data Source

[https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es](https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es)