---
layout: model
title: Indonesian DistilBERT Embeddings
author: John Snow Labs
name: distilbert_embeddings_distilbert_base_indonesian
date: 2022-04-12
tags: [distilbert, embeddings, id, open_source]
task: Embeddings
language: id
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
recommended: true
annotator: DistilBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBERT Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `distilbert-base-indonesian` is a Indonesian model orginally trained by `cahya`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_distilbert_base_indonesian_id_3.4.2_3.0_1649783905452.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_distilbert_base_indonesian","id") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Saya suka percikan NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_distilbert_base_indonesian","id") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Saya suka percikan NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("id.embed.distilbert").predict("""Saya suka percikan NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_embeddings_distilbert_base_indonesian|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|id|
|Size:|253.4 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/cahya/distilbert-base-indonesian
- https://github.com/cahya-wirawan/indonesian-language-models/tree/master/Transformers
