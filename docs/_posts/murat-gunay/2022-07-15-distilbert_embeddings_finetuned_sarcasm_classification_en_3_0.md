---
layout: model
title: English DistilBERT Embeddings Cased model (from mrm8488)
author: John Snow Labs
name: distilbert_embeddings_finetuned_sarcasm_classification
date: 2022-07-15
tags: [open_source, distilbert, embeddings, sarcasm, en]
task: Embeddings
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: DistilBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBERT Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilbert_embeddings_finetuned_sarcasm_classification` is a English model originally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_finetuned_sarcasm_classification_en_4.0.0_3.0_1657884379182.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_finetuned_sarcasm_classification_en_4.0.0_3.0_1657884379182.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_finetuned_sarcasm_classification","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["PUT YOUR STRING HERE."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_finetuned_sarcasm_classification","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("PUT YOUR STRING HERE.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_embeddings_finetuned_sarcasm_classification|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|247.6 MB|
|Case sensitive:|false|

## References

https://huggingface.co/mrm8488/distilbert-finetuned-sarcasm-classification