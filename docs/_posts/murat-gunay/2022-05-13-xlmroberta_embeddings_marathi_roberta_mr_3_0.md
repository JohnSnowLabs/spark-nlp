---
layout: model
title: Marathi XLMRoBerta Embeddings (from l3cube-pune)
author: John Snow Labs
name: xlmroberta_embeddings_marathi_roberta
date: 2022-05-13
tags: [mr, open_source, xlm_roberta, embeddings]
task: Embeddings
language: mr
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: XlmRoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XLMRoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `marathi-roberta` is a Marathi model orginally trained by `l3cube-pune`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_embeddings_marathi_roberta_mr_3.4.4_3.0_1652440151547.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
  
embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_marathi_roberta","mr") \
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

val embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_marathi_roberta","mr") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("मला स्पार्क एनएलपी आवडते").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_embeddings_marathi_roberta|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|mr|
|Size:|1.0 GB|
|Case sensitive:|true|

## References

- https://huggingface.co/l3cube-pune/marathi-roberta
- https://github.com/l3cube-pune/MarathiNLP
- https://arxiv.org/abs/2202.01159