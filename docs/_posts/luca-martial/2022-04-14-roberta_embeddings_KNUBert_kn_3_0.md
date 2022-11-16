---
layout: model
title: Kannada RoBERTa Embeddings (from Chakita)
author: John Snow Labs
name: roberta_embeddings_KNUBert
date: 2022-04-14
tags: [roberta, embeddings, kn, open_source]
task: Embeddings
language: kn
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `KNUBert` is a Kannada model orginally trained by `Chakita`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_KNUBert_kn_3.4.2_3.0_1649948307214.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols("document") \
.setOutputCol("token")

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_KNUBert","kn") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["ನಾನು ಸ್ಪಾರ್ಕ್ ಎನ್ಎಲ್ಪಿ ಪ್ರೀತಿಸುತ್ತೇನೆ"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_KNUBert","kn") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("ನಾನು ಸ್ಪಾರ್ಕ್ ಎನ್ಎಲ್ಪಿ ಪ್ರೀತಿಸುತ್ತೇನೆ").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("kn.embed.KNUBert").predict("""ನಾನು ಸ್ಪಾರ್ಕ್ ಎನ್ಎಲ್ಪಿ ಪ್ರೀತಿಸುತ್ತೇನೆ""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_KNUBert|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|kn|
|Size:|314.5 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/Chakita/KNUBert