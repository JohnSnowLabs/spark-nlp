---
layout: model
title: English pretrained_clickbait_1078k_banglabert_base_epoch_10 BertEmbeddings from samanjoy2
author: John Snow Labs
name: pretrained_clickbait_1078k_banglabert_base_epoch_10
date: 2025-01-28
tags: [en, open_source, onnx, embeddings, bert]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`pretrained_clickbait_1078k_banglabert_base_epoch_10` is a English model originally trained by samanjoy2.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pretrained_clickbait_1078k_banglabert_base_epoch_10_en_5.5.1_3.0_1738035479262.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pretrained_clickbait_1078k_banglabert_base_epoch_10_en_5.5.1_3.0_1738035479262.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("pretrained_clickbait_1078k_banglabert_base_epoch_10","en") \
      .setInputCols(["document", "token"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, tokenizer, embeddings])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("pretrained_clickbait_1078k_banglabert_base_epoch_10","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pretrained_clickbait_1078k_banglabert_base_epoch_10|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|412.1 MB|

## References

https://huggingface.co/samanjoy2/pretrained_clickbait_1078k_banglabert_base_epoch_10