---
layout: model
title: Lemmatizer (Dutch, SpacyLookup)
author: John Snow Labs
name: lemma_spacylookup
date: 2022-03-03
tags: [open_source, lemmatizer, nl]
task: Lemmatization
language: nl
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Dutch Lemmatizer is an scalable, production-ready version of the Rule-based Lemmatizer available in [Spacy Lookups Data repository](https://github.com/explosion/spacy-lookups-data/).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_spacylookup_nl_3.4.1_3.0_1646316580197.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

lemmatizer = LemmatizerModel.pretrained("lemma_spacylookup","nl") \
.setInputCols(["token"]) \
.setOutputCol("lemma")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, lemmatizer]) 

example = spark.createDataFrame([["Je bent niet beter dan ik"]], ["text"]) 

results = pipeline.fit(example).transform(example)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document")) 
.setOutputCol("token")

val lemmatizer = LemmatizerModel.pretrained("lemma_spacylookup","nl") 
.setInputCols(Array("token")) 
.setOutputCol("lemma")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, lemmatizer))
val data = Seq("Je bent niet beter dan ik").toDF("text")
val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("nl.lemma").predict("""Je bent niet beter dan ik""")
```

</div>

## Results

```bash
+--------------------------------+
|result                          |
+--------------------------------+
|[Je, bent, niet, beter, dan, ik]|
+--------------------------------+

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
|Language:|nl|
|Size:|2.5 MB|