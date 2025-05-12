---
layout: model
title: Multilingual nuner_multilingual_v0_1_finetuned_ner_pipeline pipeline BertForTokenClassification from sindhujag26
author: John Snow Labs
name: nuner_multilingual_v0_1_finetuned_ner_pipeline
date: 2025-02-06
tags: [xx, open_source, pipeline, onnx]
task: Named Entity Recognition
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nuner_multilingual_v0_1_finetuned_ner_pipeline` is a Multilingual model originally trained by sindhujag26.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nuner_multilingual_v0_1_finetuned_ner_pipeline_xx_5.5.1_3.0_1738842146047.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nuner_multilingual_v0_1_finetuned_ner_pipeline_xx_5.5.1_3.0_1738842146047.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nuner_multilingual_v0_1_finetuned_ner_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nuner_multilingual_v0_1_finetuned_ner_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nuner_multilingual_v0_1_finetuned_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|665.1 MB|

## References

https://huggingface.co/sindhujag26/NuNER-multilingual-v0.1-finetuned-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification