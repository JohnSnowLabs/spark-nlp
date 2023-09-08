---
layout: model
title: English cross_all_mpnet_base_v2_finetuned_webnlg2020_metric_average MPNetEmbeddings from teven
author: John Snow Labs
name: cross_all_mpnet_base_v2_finetuned_webnlg2020_metric_average
date: 2023-09-07
tags: [mpnet, en, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.1.1
spark_version: 3.0
supported: true
engine: onnx
annotator: MPNetEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cross_all_mpnet_base_v2_finetuned_webnlg2020_metric_average` is a English model originally trained by teven.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cross_all_mpnet_base_v2_finetuned_webnlg2020_metric_average_en_5.1.1_3.0_1694128061507.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cross_all_mpnet_base_v2_finetuned_webnlg2020_metric_average_en_5.1.1_3.0_1694128061507.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
embeddings =MPNetEmbeddings.pretrained("cross_all_mpnet_base_v2_finetuned_webnlg2020_metric_average","en") \
            .setInputCols(["documents"]) \
            .setOutputCol("mpnet_embeddings")

pipeline = Pipeline().setStages([document_assembler, embeddings])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("documents")
    
val embeddings = MPNetEmbeddings 
    .pretrained("cross_all_mpnet_base_v2_finetuned_webnlg2020_metric_average", "en")
    .setInputCols(Array("documents")) 
    .setOutputCol("mpnet_embeddings") 

val pipeline = new Pipeline().setStages(Array(document_assembler, embeddings))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cross_all_mpnet_base_v2_finetuned_webnlg2020_metric_average|
|Compatibility:|Spark NLP 5.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[mpnet_embeddings]|
|Language:|en|
|Size:|406.9 MB|

## References

https://huggingface.co/teven/cross_all-mpnet-base-v2_finetuned_WebNLG2020_metric_average