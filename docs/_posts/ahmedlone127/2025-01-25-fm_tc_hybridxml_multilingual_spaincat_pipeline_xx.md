---
layout: model
title: Multilingual fm_tc_hybridxml_multilingual_spaincat_pipeline pipeline XlmRoBertaForSequenceClassification from adriansanz
author: John Snow Labs
name: fm_tc_hybridxml_multilingual_spaincat_pipeline
date: 2025-01-25
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fm_tc_hybridxml_multilingual_spaincat_pipeline` is a Multilingual model originally trained by adriansanz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fm_tc_hybridxml_multilingual_spaincat_pipeline_xx_5.5.1_3.0_1737816990197.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fm_tc_hybridxml_multilingual_spaincat_pipeline_xx_5.5.1_3.0_1737816990197.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fm_tc_hybridxml_multilingual_spaincat_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fm_tc_hybridxml_multilingual_spaincat_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fm_tc_hybridxml_multilingual_spaincat_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|780.1 MB|

## References

https://huggingface.co/adriansanz/fm-tc-hybridXML-MULTILINGUAL-spaincat

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification