---
layout: model
title: Marathi Bert Embeddings
author: John Snow Labs
name: bert_embeddings_marathi_bert
date: 2022-04-11
tags: [bert, embeddings, mr, open_source]
task: Embeddings
language: mr
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
recommended: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `marathi-bert` is a Marathi model orginally trained by `l3cube-pune`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_marathi_bert_mr_3.4.2_3.0_1649675042393.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_marathi_bert","mr") \
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

val embeddings = BertEmbeddings.pretrained("bert_embeddings_marathi_bert","mr") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("मला स्पार्क एनएलपी आवडते").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("mr.embed.marathi_bert").predict("""मला स्पार्क एनएलपी आवडते""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_marathi_bert|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|mr|
|Size:|668.0 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/l3cube-pune/marathi-bert
- https://github.com/l3cube-pune/MarathiNLP
- https://arxiv.org/abs/2202.01159
