---
layout: model
title: English fs_distilroberta_fine_tuned_pipeline pipeline RoBertaForSequenceClassification from FinScience
author: John Snow Labs
name: fs_distilroberta_fine_tuned_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fs_distilroberta_fine_tuned_pipeline` is a English model originally trained by FinScience.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fs_distilroberta_fine_tuned_pipeline_en_5.4.2_3.0_1725167371898.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fs_distilroberta_fine_tuned_pipeline_en_5.4.2_3.0_1725167371898.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fs_distilroberta_fine_tuned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fs_distilroberta_fine_tuned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fs_distilroberta_fine_tuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|308.8 MB|

## References

https://huggingface.co/FinScience/FS-distilroberta-fine-tuned

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification