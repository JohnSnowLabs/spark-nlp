---
layout: model
title: English danskbert_fgn_yemen2016_pipeline pipeline XlmRoBertaForSequenceClassification from yemen2016
author: John Snow Labs
name: danskbert_fgn_yemen2016_pipeline
date: 2024-12-17
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`danskbert_fgn_yemen2016_pipeline` is a English model originally trained by yemen2016.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/danskbert_fgn_yemen2016_pipeline_en_5.5.1_3.0_1734417865881.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/danskbert_fgn_yemen2016_pipeline_en_5.5.1_3.0_1734417865881.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("danskbert_fgn_yemen2016_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("danskbert_fgn_yemen2016_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|danskbert_fgn_yemen2016_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|423.2 MB|

## References

https://huggingface.co/yemen2016/danskbert-FGN

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification