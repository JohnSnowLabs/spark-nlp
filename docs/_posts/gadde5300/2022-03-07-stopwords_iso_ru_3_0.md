---
layout: model
title: Stopwords Remover for Russian language (768 entries)
author: John Snow Labs
name: stopwords_iso
date: 2022-03-07
tags: [stopwords, ru, open_source]
task: Stop Words Removal
language: ru
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a scalable, production-ready Stopwords Remover model trained using the corpus available at [stopwords-iso](https://github.com/stopwords-iso/).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_iso_ru_3.4.1_3.0_1646672321516.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

stop_words = StopWordsCleaner.pretrained("stopwords_iso","ru") \
.setInputCols(["token"]) \
.setOutputCol("cleanTokens")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, stop_words]) 

example = spark.createDataFrame([["Ты не лучше меня"]], ["text"]) 

results = pipeline.fit(example).transform(example)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val stop_words = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val lemmatizer = StopWordsCleaner.pretrained("stopwords_iso","ru") 
.setInputCols(Array("token")) 
.setOutputCol("cleanTokens")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, stop_words))
val data = Seq("Ты не лучше меня").toDF("text")
val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ru.stopwords").predict("""Ты не лучше меня""")
```

</div>

## Results

```bash
+-------+
|result |
+-------+
|[лучше]|
+-------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_iso|
|Compatibility:|Spark NLP 3.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|ru|
|Size:|4.1 KB|