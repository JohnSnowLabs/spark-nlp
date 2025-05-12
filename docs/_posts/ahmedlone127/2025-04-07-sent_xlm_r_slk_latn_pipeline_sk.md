---
layout: model
title: Slovak sent_xlm_r_slk_latn_pipeline pipeline XlmRoBertaSentenceEmbeddings from DGurgurov
author: John Snow Labs
name: sent_xlm_r_slk_latn_pipeline
date: 2025-04-07
tags: [sk, open_source, pipeline, onnx]
task: Embeddings
language: sk
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_xlm_r_slk_latn_pipeline` is a Slovak model originally trained by DGurgurov.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_xlm_r_slk_latn_pipeline_sk_5.5.1_3.0_1744024916775.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_xlm_r_slk_latn_pipeline_sk_5.5.1_3.0_1744024916775.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_xlm_r_slk_latn_pipeline", lang = "sk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_xlm_r_slk_latn_pipeline", lang = "sk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_xlm_r_slk_latn_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|sk|
|Size:|1.0 GB|

## References

https://huggingface.co/DGurgurov/xlm-r_slk-latn

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- XlmRoBertaSentenceEmbeddings