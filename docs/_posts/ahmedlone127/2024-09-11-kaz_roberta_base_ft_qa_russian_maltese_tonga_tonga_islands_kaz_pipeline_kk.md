---
layout: model
title: Kazakh kaz_roberta_base_ft_qa_russian_maltese_tonga_tonga_islands_kaz_pipeline pipeline RoBertaForQuestionAnswering from med-alex
author: John Snow Labs
name: kaz_roberta_base_ft_qa_russian_maltese_tonga_tonga_islands_kaz_pipeline
date: 2024-09-11
tags: [kk, open_source, pipeline, onnx]
task: Question Answering
language: kk
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kaz_roberta_base_ft_qa_russian_maltese_tonga_tonga_islands_kaz_pipeline` is a Kazakh model originally trained by med-alex.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kaz_roberta_base_ft_qa_russian_maltese_tonga_tonga_islands_kaz_pipeline_kk_5.5.0_3.0_1726039496181.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kaz_roberta_base_ft_qa_russian_maltese_tonga_tonga_islands_kaz_pipeline_kk_5.5.0_3.0_1726039496181.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kaz_roberta_base_ft_qa_russian_maltese_tonga_tonga_islands_kaz_pipeline", lang = "kk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kaz_roberta_base_ft_qa_russian_maltese_tonga_tonga_islands_kaz_pipeline", lang = "kk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kaz_roberta_base_ft_qa_russian_maltese_tonga_tonga_islands_kaz_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|kk|
|Size:|311.7 MB|

## References

https://huggingface.co/med-alex/kaz-roberta-base-ft-qa-ru-mt-to-kaz

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering