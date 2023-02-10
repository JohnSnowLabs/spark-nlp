---
layout: model
title: Stopwords Remover for Malayalam language (9 entries)
author: John Snow Labs
name: stopwords_iso
date: 2022-03-07
tags: [stopwords, ml, open_source]
task: Stop Words Removal
language: ml
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_iso_ml_3.4.1_3.0_1646672962533.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/stopwords_iso_ml_3.4.1_3.0_1646672962533.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

stop_words = StopWordsCleaner.pretrained("stopwords_iso","ml") \
.setInputCols(["token"]) \
.setOutputCol("cleanTokens")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, stop_words]) 

example = spark.createDataFrame([["നിങ്ങൾ എന്നെക്കാൾ മികച്ചവനല്ല"]], ["text"]) 

results = pipeline.fit(example).transform(example)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val stop_words = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val lemmatizer = StopWordsCleaner.pretrained("stopwords_iso","ml") 
.setInputCols(Array("token")) 
.setOutputCol("cleanTokens")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, stop_words))
val data = Seq("നിങ്ങൾ എന്നെക്കാൾ മികച്ചവനല്ല").toDF("text")
val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ml.stopwords").predict("""നിങ്ങൾ എന്നെക്കാൾ മികച്ചവനല്ല""")
```

</div>

## Results

```bash
+---------------------------------+
|result                           |
+---------------------------------+
|[നിങ്ങൾ, എന്നെക്കാൾ, മികച്ചവനല്ല]|
+---------------------------------+

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
|Language:|ml|
|Size:|1.4 KB|