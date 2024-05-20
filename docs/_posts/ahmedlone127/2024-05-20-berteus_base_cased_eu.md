---
layout: model
title: Basque berteus_base_cased BertEmbeddings from ixa-ehu
author: John Snow Labs
name: berteus_base_cased
date: 2024-05-20
tags: [eu, open_source, embeddings, bert, onnx]
task: Embeddings
language: eu
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`berteus_base_cased` is a Basque model originally trained by ixa-ehu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/berteus_base_cased_eu_5.2.4_3.0_1716221783534.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/berteus_base_cased_eu_5.2.4_3.0_1716221783534.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("berteus_base_cased","eu") \
      .setInputCols(["document", "token"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, tokenizer, embeddings])
data = spark.createDataFrame([["Saya suka Spark NLP"]]).toDF("text")
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

val embeddings = BertEmbeddings.pretrained("berteus_base_cased","eu") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))
val data = Seq("Où est-ce que je vis?","Mon nom est Wolfgang et je vis à Berlin.").toDS.toDF("document_question", "document_context")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|berteus_base_cased|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[bert]|
|Language:|eu|
|Size:|464.6 MB|

## References

https://huggingface.co/ixa-ehu/berteus-base-cased