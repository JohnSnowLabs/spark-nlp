---
layout: model
title: English indic_bert_kornesh AlbertEmbeddings from kornesh
author: John Snow Labs
name: indic_bert_kornesh
date: 2024-05-20
tags: [en, open_source, embeddings, albert, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: AlbertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indic_bert_kornesh` is a English model originally trained by kornesh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indic_bert_kornesh_en_5.2.4_3.0_1716202404783.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indic_bert_kornesh_en_5.2.4_3.0_1716202404783.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


val documentAssembler = new DocumentAssembler() 
        .setInputCol("text") 
        .setOutputCol("document")
    
val tokenizer = new Tokenizer() 
        .setInputCols(Array("document"))
        .setOutputCol("token")

val embeddings = AlbertEmbeddings.pretrained("indic_bert_kornesh","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))
val data = Seq("Où est-ce que je vis?","Mon nom est Wolfgang et je vis à Berlin.").toDS.toDF("document_question", "document_context")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
```scala
            
documentAssembler = DocumentAssembler()         .setInputCol("text")         .setOutputCol("document")
    
tokenizer = Tokenizer()         .setInputCols("document")         .setOutputCol("token")

embeddings = AlbertEmbeddings.pretrained("indic_bert_kornesh","en")         .setInputCols(["document", "token"])         .setOutputCol("embeddings")
        
        
pipeline = Pipeline().setStages([documentAssembler, tokenizer, embeddings])
data = spark.createDataFrame([["Saya suka Spark NLP"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indic_bert_kornesh|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[albert]|
|Language:|en|
|Size:|125.5 MB|

## References

https://huggingface.co/kornesh/indic-bert