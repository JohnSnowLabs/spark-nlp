---
layout: model
title: Malay (macrolanguage) sent_melayubert_pipeline pipeline BertSentenceEmbeddings from StevenLimcorn
author: John Snow Labs
name: sent_melayubert_pipeline
date: 2024-09-25
tags: [ms, open_source, pipeline, onnx]
task: Embeddings
language: ms
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_melayubert_pipeline` is a Malay (macrolanguage) model originally trained by StevenLimcorn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_melayubert_pipeline_ms_5.5.0_3.0_1727252961221.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_melayubert_pipeline_ms_5.5.0_3.0_1727252961221.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_melayubert_pipeline", lang = "ms")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_melayubert_pipeline", lang = "ms")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_melayubert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ms|
|Size:|408.7 MB|

## References

https://huggingface.co/StevenLimcorn/MelayuBERT

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings