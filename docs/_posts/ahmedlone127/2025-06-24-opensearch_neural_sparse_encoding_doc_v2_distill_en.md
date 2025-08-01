---
layout: model
title: English opensearch_neural_sparse_encoding_doc_v2_distill DistilBertEmbeddings from opensearch-project
author: John Snow Labs
name: opensearch_neural_sparse_encoding_doc_v2_distill
date: 2025-06-24
tags: [en, open_source, onnx, embeddings, distilbert, openvino]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: DistilBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`opensearch_neural_sparse_encoding_doc_v2_distill` is a English model originally trained by opensearch-project.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opensearch_neural_sparse_encoding_doc_v2_distill_en_5.5.1_3.0_1750780394262.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/opensearch_neural_sparse_encoding_doc_v2_distill_en_5.5.1_3.0_1750780394262.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DistilBertEmbeddings.pretrained("opensearch_neural_sparse_encoding_doc_v2_distill","en") \
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

val embeddings = DistilBertEmbeddings.pretrained("opensearch_neural_sparse_encoding_doc_v2_distill","en") 
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
|Model Name:|opensearch_neural_sparse_encoding_doc_v2_distill|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[distilbert]|
|Language:|en|
|Size:|247.3 MB|

## References

References

https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill