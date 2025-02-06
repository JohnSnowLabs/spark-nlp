---
layout: model
title: Mongolian roberta_base_ner_demo_fire12_pipeline pipeline RoBertaForTokenClassification from fire12
author: John Snow Labs
name: roberta_base_ner_demo_fire12_pipeline
date: 2025-02-06
tags: [mn, open_source, pipeline, onnx]
task: Named Entity Recognition
language: mn
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_ner_demo_fire12_pipeline` is a Mongolian model originally trained by fire12.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_ner_demo_fire12_pipeline_mn_5.5.1_3.0_1738882768084.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_ner_demo_fire12_pipeline_mn_5.5.1_3.0_1738882768084.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_ner_demo_fire12_pipeline", lang = "mn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_ner_demo_fire12_pipeline", lang = "mn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_ner_demo_fire12_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|mn|
|Size:|465.7 MB|

## References

https://huggingface.co/fire12/roberta-base-ner-demo

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification