---
layout: model
title: Ukrainian instruction_detection_pipeline pipeline XlmRoBertaForTokenClassification from zeusfsx
author: John Snow Labs
name: instruction_detection_pipeline
date: 2024-09-10
tags: [uk, open_source, pipeline, onnx]
task: Named Entity Recognition
language: uk
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`instruction_detection_pipeline` is a Ukrainian model originally trained by zeusfsx.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/instruction_detection_pipeline_uk_5.5.0_3.0_1726011536218.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/instruction_detection_pipeline_uk_5.5.0_3.0_1726011536218.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("instruction_detection_pipeline", lang = "uk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("instruction_detection_pipeline", lang = "uk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|instruction_detection_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|uk|
|Size:|390.6 MB|

## References

https://huggingface.co/zeusfsx/instruction-detection

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification