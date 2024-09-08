---
layout: model
title: English cs4248_roberta_wolof_search_mix_epoch_3_pipeline pipeline RoBertaForQuestionAnswering from BenjaminLHR
author: John Snow Labs
name: cs4248_roberta_wolof_search_mix_epoch_3_pipeline
date: 2024-09-04
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cs4248_roberta_wolof_search_mix_epoch_3_pipeline` is a English model originally trained by BenjaminLHR.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cs4248_roberta_wolof_search_mix_epoch_3_pipeline_en_5.5.0_3.0_1725484266762.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cs4248_roberta_wolof_search_mix_epoch_3_pipeline_en_5.5.0_3.0_1725484266762.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cs4248_roberta_wolof_search_mix_epoch_3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cs4248_roberta_wolof_search_mix_epoch_3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cs4248_roberta_wolof_search_mix_epoch_3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|465.9 MB|

## References

https://huggingface.co/BenjaminLHR/cs4248-roberta-wo-search-mix-epoch-3

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering