---
layout: model
title: English Legal BERT Embeddings (from dlicari)
author: John Snow Labs
name: bert_embeddings_lsg16k_Italian_Legal
date: 2023-02-21
tags: [longformer, it, italian, embeddings, transformer, open_source, tensorflow]
task: Embeddings
language: it
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `lsg16k-Italian-Legal-BERT` is a Italian model originally trained by `dlicari`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_lsg16k_Italian_Legal_it_4.3.0_3.0_1676980225424.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_lsg16k_Italian_Legal_it_4.3.0_3.0_1676980225424.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 documentAssembler = nlp.DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

tokenizer = nlp.Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")
  
embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_lsg16k_Italian_Legal","it") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = nlp.Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Adoro Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_lsg16k_Italian_Legal","it") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Adoro Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_lsg16k_Italian_Legal|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|it|
|Size:|457.4 MB|
|Case sensitive:|true|

## References

https://huggingface.co/dlicari/lsg16k-Italian-Legal-BERT