---
layout: model
title: English inisw08_distilbert_mlm_adamw_torch_fused DistilBertEmbeddings from ugiugi
author: John Snow Labs
name: inisw08_distilbert_mlm_adamw_torch_fused
date: 2023-09-15
tags: [distilbert, en, open_source, fill_mask, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.1.2
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`inisw08_distilbert_mlm_adamw_torch_fused` is a English model originally trained by ugiugi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/inisw08_distilbert_mlm_adamw_torch_fused_en_5.1.2_3.0_1694788244866.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/inisw08_distilbert_mlm_adamw_torch_fused_en_5.1.2_3.0_1694788244866.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
embeddings =DistilBertEmbeddings.pretrained("inisw08_distilbert_mlm_adamw_torch_fused","en") \
            .setInputCols(["documents","token"]) \
            .setOutputCol("embeddings")

pipeline = Pipeline().setStages([document_assembler, embeddings])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("embeddings")
    
val embeddings = DistilBertEmbeddings 
    .pretrained("inisw08_distilbert_mlm_adamw_torch_fused", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("embeddings") 

val pipeline = new Pipeline().setStages(Array(document_assembler, embeddings))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|inisw08_distilbert_mlm_adamw_torch_fused|
|Compatibility:|Spark NLP 5.1.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/ugiugi/inisw08-DistilBERT-mlm-adamw_torch_fused