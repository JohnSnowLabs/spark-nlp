---
layout: model
title: Lemmatizer (Serbian)
author: John Snow Labs
name: lemma_spacylookup
date: 2022-03-03
tags: [lemma, open_source, sr]
task: Lemmatization
language: sr
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Serbian Lemmatizer is an scalable, production-ready version of the Rule-based lemmatizer available in [Spacy](https://github.com/explosion/spaCy/tree/master/spacy/lang)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_spacylookup_sr_3.4.1_3.0_1646306220303.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

    lemmatizer = LemmatizerModel.pretrained("lemma_spacylookup","ca") \
        .setInputCols(["token"]) \
        .setOutputCol("lemma")

    pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer]) 

    example = spark.createDataFrame([["No ets millor que jo"]], ["text"]) 

    results = pipeline.fit(example).transform(example)
```
```scala
val documentAssembler = DocumentAssembler() 
            .setInputCol("text") 
            .setOutputCol("document")

    val tokenizer = Tokenizer() 
        .setInputCols(Array("sentence")) 
        .setOutputCol("token")

    val lemmatizer = LemmatizerModel.pretrained("lemma_spacylookup","a") 
        .setInputCols(Array("token")) 
        .setOutputCol("lemma")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, lemmatizer))
    val data = Seq("i am f").toDF("text")
    val results = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+--------------------------+
|result                    |
+--------------------------+
|[No, ets, millor, que, jo]|
+--------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|lemma_spacylookup|
|Compatibility:|Spark NLP 3.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[lemma]|
|Language:|sr|
|Size:|7.0 MB|