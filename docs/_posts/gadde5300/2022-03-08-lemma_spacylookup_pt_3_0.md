---
layout: model
title: Lemmatizer (Portugese, SpacyLookup)
author: John Snow Labs
name: lemma_spacylookup
date: 2022-03-08
tags: [open_source, lemmatizer, pt]
task: Lemmatization
language: pt
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Portugese Lemmatizer is an scalable, production-ready version of the Rule-based Lemmatizer available in [Spacy Lookups Data repository](https://github.com/explosion/spacy-lookups-data/).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_spacylookup_pt_3.4.1_3.0_1646753629532.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

lemmatizer = LemmatizerModel.pretrained("lemma_spacylookup","pt") \
.setInputCols(["token"]) \
.setOutputCol("lemma")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, lemmatizer]) 

example = spark.createDataFrame([["Você não é melhor que eu"]], ["text"]) 

results = pipeline.fit(example).transform(example)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val lemmatizer = LemmatizerModel.pretrained("lemma_spacylookup","pt") 
.setInputCols(Array("token")) 
.setOutputCol("lemma")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, lemmatizer))
val data = Seq("Você não é melhor que eu").toDF("text")
val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("pt.lemma.spacylookup").predict("""Você não é melhor que eu""")
```

</div>

## Results

```bash
+---------------------------------+
|result                           |
+---------------------------------+
|[Você, não, ser, melhor, que, eu]|
+---------------------------------+

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
|Language:|pt|
|Size:|8.9 MB|