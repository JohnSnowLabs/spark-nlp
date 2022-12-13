---
layout: model
title: Marathi ALBERT Embeddings (v2)
author: John Snow Labs
name: albert_embeddings_marathi_albert_v2
date: 2022-04-14
tags: [albert, embeddings, mr, open_source]
task: Embeddings
language: mr
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: AlBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained ALBERT Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `marathi-albert-v2` is a Marathi model orginally trained by `l3cube-pune`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_embeddings_marathi_albert_v2_mr_3.4.2_3.0_1649954249563.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_embeddings_marathi_albert_v2_mr_3.4.2_3.0_1649954249563.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = AlbertEmbeddings.pretrained("albert_embeddings_marathi_albert_v2","mr") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["मला स्पार्क एनएलपी आवडते"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = AlbertEmbeddings.pretrained("albert_embeddings_marathi_albert_v2","mr") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("मला स्पार्क एनएलपी आवडते").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("mr.embed.albert_v2").predict("""मला स्पार्क एनएलपी आवडते""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_embeddings_marathi_albert_v2|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|mr|
|Size:|128.3 MB|
|Case sensitive:|false|

## References

- https://huggingface.co/l3cube-pune/marathi-albert-v2
- https://github.com/l3cube-pune/MarathiNLP
- https://arxiv.org/abs/2202.01159