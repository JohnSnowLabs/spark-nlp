---
layout: model
title: English babyberta_aochildes_2_5m_wikipedia1_2_5m_with_masking_seed6_finetuned_squad_pipeline pipeline RoBertaForQuestionAnswering from lielbin
author: John Snow Labs
name: babyberta_aochildes_2_5m_wikipedia1_2_5m_with_masking_seed6_finetuned_squad_pipeline
date: 2024-09-12
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`babyberta_aochildes_2_5m_wikipedia1_2_5m_with_masking_seed6_finetuned_squad_pipeline` is a English model originally trained by lielbin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/babyberta_aochildes_2_5m_wikipedia1_2_5m_with_masking_seed6_finetuned_squad_pipeline_en_5.5.0_3.0_1726176098952.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/babyberta_aochildes_2_5m_wikipedia1_2_5m_with_masking_seed6_finetuned_squad_pipeline_en_5.5.0_3.0_1726176098952.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("babyberta_aochildes_2_5m_wikipedia1_2_5m_with_masking_seed6_finetuned_squad_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("babyberta_aochildes_2_5m_wikipedia1_2_5m_with_masking_seed6_finetuned_squad_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|babyberta_aochildes_2_5m_wikipedia1_2_5m_with_masking_seed6_finetuned_squad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|32.0 MB|

## References

https://huggingface.co/lielbin/BabyBERTa-aochildes_2.5M_wikipedia1_2.5M-with-Masking-seed6-finetuned-SQuAD

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering