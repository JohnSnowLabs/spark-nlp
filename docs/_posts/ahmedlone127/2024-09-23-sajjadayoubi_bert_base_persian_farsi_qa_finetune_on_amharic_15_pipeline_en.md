---
layout: model
title: English sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15_pipeline pipeline BertForQuestionAnswering from phd411r1
author: John Snow Labs
name: sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15_pipeline
date: 2024-09-23
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15_pipeline` is a English model originally trained by phd411r1.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15_pipeline_en_5.5.0_3.0_1727128061168.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15_pipeline_en_5.5.0_3.0_1727128061168.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|606.5 MB|

## References

https://huggingface.co/phd411r1/SajjadAyoubi_bert-base-fa-qa_finetune_on_am_15

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering