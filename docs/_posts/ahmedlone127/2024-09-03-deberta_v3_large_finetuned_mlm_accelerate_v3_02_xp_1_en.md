---
layout: model
title: English deberta_v3_large_finetuned_mlm_accelerate_v3_02_xp_1 DeBertaEmbeddings from quastrinos
author: John Snow Labs
name: deberta_v3_large_finetuned_mlm_accelerate_v3_02_xp_1
date: 2024-09-03
tags: [en, open_source, onnx, embeddings, deberta]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: DeBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deberta_v3_large_finetuned_mlm_accelerate_v3_02_xp_1` is a English model originally trained by quastrinos.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v3_large_finetuned_mlm_accelerate_v3_02_xp_1_en_5.5.0_3.0_1725376521475.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_v3_large_finetuned_mlm_accelerate_v3_02_xp_1_en_5.5.0_3.0_1725376521475.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DeBertaEmbeddings.pretrained("deberta_v3_large_finetuned_mlm_accelerate_v3_02_xp_1","en") \
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

val embeddings = DeBertaEmbeddings.pretrained("deberta_v3_large_finetuned_mlm_accelerate_v3_02_xp_1","en") 
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
|Model Name:|deberta_v3_large_finetuned_mlm_accelerate_v3_02_xp_1|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[deberta]|
|Language:|en|
|Size:|1.6 GB|

## References

https://huggingface.co/quastrinos/deberta-v3-large-finetuned-mlm-accelerate-v3-02-xp-1