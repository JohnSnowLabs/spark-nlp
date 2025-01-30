---
layout: model
title: English pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster GPT2Transformer from jvelja
author: John Snow Labs
name: pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster
date: 2025-01-30
tags: [en, open_source, onnx, text_generation, gpt2]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: GPT2Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster` is a English model originally trained by jvelja.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster_en_5.5.1_3.0_1738262817791.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster_en_5.5.1_3.0_1738262817791.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

seq2seq = GPT2Transformer.pretrained("pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster","en") \
      .setInputCols(["documents"]) \
      .setOutputCol("generation")       
        
pipeline = Pipeline().setStages([documentAssembler, seq2seq])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val seq2seq = GPT2Transformer.pretrained("pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster","en") 
    .setInputCols(Array("documents")) 
    .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, seq2seq))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[generation]|
|Language:|en|
|Size:|469.7 MB|

## References

https://huggingface.co/jvelja/pythia-finetune-gpt2-clusters-4-NoBSGC-lr_0.0005-Modularity-RAVEL_MIXEDFixCluster