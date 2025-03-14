---
layout: model
title: Turkish sent_convbert_base_turkish_cased_offensive_mlm_pipeline pipeline BertSentenceEmbeddings from Overfit-GM
author: John Snow Labs
name: sent_convbert_base_turkish_cased_offensive_mlm_pipeline
date: 2025-01-26
tags: [tr, open_source, pipeline, onnx]
task: Embeddings
language: tr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_convbert_base_turkish_cased_offensive_mlm_pipeline` is a Turkish model originally trained by Overfit-GM.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_convbert_base_turkish_cased_offensive_mlm_pipeline_tr_5.5.1_3.0_1737900470737.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_convbert_base_turkish_cased_offensive_mlm_pipeline_tr_5.5.1_3.0_1737900470737.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_convbert_base_turkish_cased_offensive_mlm_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_convbert_base_turkish_cased_offensive_mlm_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_convbert_base_turkish_cased_offensive_mlm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|400.6 MB|

## References

https://huggingface.co/Overfit-GM/convbert-base-turkish-cased-offensive-mlm

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings