---
layout: model
title: Russian xlm_roberta_fine_tuned_on_russian_abusive_language_pipeline pipeline XlmRoBertaForSequenceClassification from marianna13
author: John Snow Labs
name: xlm_roberta_fine_tuned_on_russian_abusive_language_pipeline
date: 2024-09-13
tags: [ru, open_source, pipeline, onnx]
task: Text Classification
language: ru
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_fine_tuned_on_russian_abusive_language_pipeline` is a Russian model originally trained by marianna13.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_fine_tuned_on_russian_abusive_language_pipeline_ru_5.5.0_3.0_1726258230332.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_fine_tuned_on_russian_abusive_language_pipeline_ru_5.5.0_3.0_1726258230332.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_fine_tuned_on_russian_abusive_language_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_fine_tuned_on_russian_abusive_language_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_fine_tuned_on_russian_abusive_language_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|1.0 GB|

## References

https://huggingface.co/marianna13/xlm-roberta-fine-tuned-on-russian-abusive-language

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification