---
layout: model
title: Public Health Surveillance (PHS) BERT Embeddings
author: John Snow Labs
name: bert_embeddings_phs_bert
date: 2022-07-02
tags: [bert, en, embeddings, open_source]
task: Embeddings
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT Embeddings model, adapted from [Hugging Face](https://huggingface.co/publichealthsurveillance/PHS-BERT) and curated to provide scalability and production-readiness using Spark NLP. `PHS-BERT` is an English model and trained to identify the tasks related to public health surveillance (PHS) on social media.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_phs_bert_en_4.0.0_3.0_1656759538082.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_phs_bert_en_4.0.0_3.0_1656759538082.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = BertEmbeddings.pretrained("bert_embeddings_phs_bert","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["No place in my city has shelter space for us, and I won't put my baby on the literal street. What cities have good shelter programs for homeless mothers and children?"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_phs_bert","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("No place in my city has shelter space for us, and I won't put my baby on the literal street. What cities have good shelter programs for homeless mothers and children?").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_phs_bert|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|false|

## References

https://arxiv.org/abs/2204.04521
