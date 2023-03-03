---
layout: model
title: English Bert Embeddings Cased model (from aditeyabaral)
author: John Snow Labs
name: bert_embeddings_carlbert_webex_mlm_spatial
date: 2023-02-22
tags: [open_source, bert, bert_embeddings, bertformaskedlm, en, tensorflow]
task: Embeddings
language: en
nav_key: models
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

Pretrained BertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `carlbert-webex-mlm-spatial` is a English model originally trained by `aditeyabaral`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_carlbert_webex_mlm_spatial_en_4.3.0_3.0_1677087512961.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_carlbert_webex_mlm_spatial_en_4.3.0_3.0_1677087512961.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

embeddings = BertEmbeddings.pretrained("bert_embeddings_carlbert_webex_mlm_spatial","de") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Ich liebe Spark-NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols(Array("text")) 
    .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val embeddings = BertEmbeddings.pretrained("bert_embeddings_carlbert_webex_mlm_spatial","de") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(True)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Ich liebe Spark-NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_carlbert_webex_mlm_spatial|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|406.6 MB|
|Case sensitive:|true|

## References

https://huggingface.co/aditeyabaral/carlbert-webex-mlm-spatial
