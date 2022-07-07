---
layout: model
title: Lemmatization from BSC/projecte_aina lookups
author: cayorodriguez
name: lemmatizer_bsc
date: 2022-07-07
tags: [ca, open_source]
task: Lemmatization
language: ca
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Lemmatizer using lookup tables from BSC/projecte_aina sources

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/cayorodriguez/lemmatizer_bsc_ca_3.4.4_3.0_1657199421685.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

lemmatizer = LemmatizerModel.pretrained("lemmatizer_bsc","ca") \
    .setInputCols(["token"]) \
    .setOutputCol("lemma")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, lemmatizer]) 

example = spark.createDataFrame([["Bons dies, al mati"]], ["text"]) 

results = pipeline.fit(example).transform(example)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|lemmatizer_bsc|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[form]|
|Output Labels:|[lemma]|
|Language:|ca|
|Size:|7.3 MB|