---
layout: model
title: English tokenizertestingmtsufall2024softwareengineering_pipeline pipeline T5Transformer from cheaptrix
author: John Snow Labs
name: tokenizertestingmtsufall2024softwareengineering_pipeline
date: 2024-12-16
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tokenizertestingmtsufall2024softwareengineering_pipeline` is a English model originally trained by cheaptrix.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tokenizertestingmtsufall2024softwareengineering_pipeline_en_5.5.1_3.0_1734327749937.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tokenizertestingmtsufall2024softwareengineering_pipeline_en_5.5.1_3.0_1734327749937.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tokenizertestingmtsufall2024softwareengineering_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tokenizertestingmtsufall2024softwareengineering_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tokenizertestingmtsufall2024softwareengineering_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|341.0 MB|

## References

https://huggingface.co/cheaptrix/TokenizerTestingMTSUFall2024SoftwareEngineering

## Included Models

- DocumentAssembler
- T5Transformer