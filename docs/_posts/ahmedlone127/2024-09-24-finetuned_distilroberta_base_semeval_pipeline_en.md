---
layout: model
title: English finetuned_distilroberta_base_semeval_pipeline pipeline RoBertaForSequenceClassification from Youssef320
author: John Snow Labs
name: finetuned_distilroberta_base_semeval_pipeline
date: 2024-09-24
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_distilroberta_base_semeval_pipeline` is a English model originally trained by Youssef320.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_distilroberta_base_semeval_pipeline_en_5.5.0_3.0_1727172137087.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_distilroberta_base_semeval_pipeline_en_5.5.0_3.0_1727172137087.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_distilroberta_base_semeval_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_distilroberta_base_semeval_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_distilroberta_base_semeval_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|308.9 MB|

## References

https://huggingface.co/Youssef320/finetuned-distilroberta-base-SemEval

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification