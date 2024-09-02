---
layout: model
title: English finetuned_token_itr0_3e_05_webdiscourse_16_02_2022_20_59_50_pipeline pipeline DistilBertForTokenClassification from ali2066
author: John Snow Labs
name: finetuned_token_itr0_3e_05_webdiscourse_16_02_2022_20_59_50_pipeline
date: 2024-09-02
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_token_itr0_3e_05_webdiscourse_16_02_2022_20_59_50_pipeline` is a English model originally trained by ali2066.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_token_itr0_3e_05_webdiscourse_16_02_2022_20_59_50_pipeline_en_5.5.0_3.0_1725267303385.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_token_itr0_3e_05_webdiscourse_16_02_2022_20_59_50_pipeline_en_5.5.0_3.0_1725267303385.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_token_itr0_3e_05_webdiscourse_16_02_2022_20_59_50_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_token_itr0_3e_05_webdiscourse_16_02_2022_20_59_50_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_token_itr0_3e_05_webdiscourse_16_02_2022_20_59_50_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/ali2066/finetuned_token_itr0_3e-05_webDiscourse_16_02_2022-20_59_50

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification