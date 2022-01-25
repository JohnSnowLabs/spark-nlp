---
layout: model
title: My Sentiment Pipeline
author: John Snow Labs
name: my_nlp_pipeline
date: 2022-01-25
tags: [sentiment, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a sentiment classifir

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/my_nlp_pipeline_en_3.4.0_3.0_1643145717972.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
|Model Name:|my_nlp_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|173.5 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- NormalizerModel
- StopWordsCleaner
- LemmatizerModel
- WordEmbeddingsModel
- SentenceEmbeddings
- ClassifierDLModel