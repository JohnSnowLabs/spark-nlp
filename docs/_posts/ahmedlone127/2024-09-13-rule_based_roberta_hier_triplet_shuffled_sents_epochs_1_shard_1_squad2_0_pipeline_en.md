---
layout: model
title: English rule_based_roberta_hier_triplet_shuffled_sents_epochs_1_shard_1_squad2_0_pipeline pipeline RoBertaForQuestionAnswering from AnonymousSub
author: John Snow Labs
name: rule_based_roberta_hier_triplet_shuffled_sents_epochs_1_shard_1_squad2_0_pipeline
date: 2024-09-13
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rule_based_roberta_hier_triplet_shuffled_sents_epochs_1_shard_1_squad2_0_pipeline` is a English model originally trained by AnonymousSub.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rule_based_roberta_hier_triplet_shuffled_sents_epochs_1_shard_1_squad2_0_pipeline_en_5.5.0_3.0_1726231864705.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rule_based_roberta_hier_triplet_shuffled_sents_epochs_1_shard_1_squad2_0_pipeline_en_5.5.0_3.0_1726231864705.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("rule_based_roberta_hier_triplet_shuffled_sents_epochs_1_shard_1_squad2_0_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("rule_based_roberta_hier_triplet_shuffled_sents_epochs_1_shard_1_squad2_0_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rule_based_roberta_hier_triplet_shuffled_sents_epochs_1_shard_1_squad2_0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|459.7 MB|

## References

https://huggingface.co/AnonymousSub/rule_based_roberta_hier_triplet_shuffled_sents_epochs_1_shard_1_squad2.0

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering