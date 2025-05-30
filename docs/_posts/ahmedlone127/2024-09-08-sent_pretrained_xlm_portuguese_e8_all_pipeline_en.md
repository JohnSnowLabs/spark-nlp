---
layout: model
title: English sent_pretrained_xlm_portuguese_e8_all_pipeline pipeline XlmRoBertaSentenceEmbeddings from harish
author: John Snow Labs
name: sent_pretrained_xlm_portuguese_e8_all_pipeline
date: 2024-09-08
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_pretrained_xlm_portuguese_e8_all_pipeline` is a English model originally trained by harish.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_pretrained_xlm_portuguese_e8_all_pipeline_en_5.5.0_3.0_1725813493514.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_pretrained_xlm_portuguese_e8_all_pipeline_en_5.5.0_3.0_1725813493514.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_pretrained_xlm_portuguese_e8_all_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_pretrained_xlm_portuguese_e8_all_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_pretrained_xlm_portuguese_e8_all_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/harish/preTrained-xlm-pt-e8-all

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- XlmRoBertaSentenceEmbeddings