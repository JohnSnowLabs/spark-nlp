---
layout: model
title: Hindi Bert Embeddings
author: John Snow Labs
name: bert_embeddings_indic_transformers_hi_bert
date: 2023-06-21
tags: [bert, embeddings, hi, open_source, onnx]
task: Embeddings
language: hi
edition: Spark NLP 5.0.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `indic-transformers-hi-bert` is a Hindi model orginally trained by `neuralspace-reverie`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_indic_transformers_hi_bert_hi_5.0.0_3.0_1687339963111.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_indic_transformers_hi_bert_hi_5.0.0_3.0_1687339963111.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



{:.model-param}

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols("document") \
.setOutputCol("token")

embeddings = BertEmbeddings.pretrained("bert_embeddings_indic_transformers_hi_bert","hi") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["मुझे स्पार्क एनएलपी पसंद है"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_indic_transformers_hi_bert","hi") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("मुझे स्पार्क एनएलपी पसंद है").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("hi.embed.indic_transformers_hi_bert").predict("""मुझे स्पार्क एनएलपी पसंद है""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_indic_transformers_hi_bert|
|Compatibility:|Spark NLP 5.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|hi|
|Size:|609.2 MB|
|Case sensitive:|true|