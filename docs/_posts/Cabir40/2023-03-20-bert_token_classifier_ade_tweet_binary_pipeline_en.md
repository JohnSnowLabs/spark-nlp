---
layout: model
title: Pipeline to Detect Adverse Drug Events (MedicalBertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ade_tweet_binary_pipeline
date: 2023-03-20
tags: [clinical, licensed, ade, en, medicalbertfortokenclassification, ner]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_token_classifier_ade_tweet_binary](https://nlp.johnsnowlabs.com/2022/07/29/bert_token_classifier_ade_tweet_binary_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ade_tweet_binary_pipeline_en_4.3.0_3.2_1679298990358.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ade_tweet_binary_pipeline_en_4.3.0_3.2_1679298990358.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ade_tweet_binary_pipeline", "en", "clinical/models")

text = '''I used to be on paxil but that made me more depressed and prozac made me angry. Maybe cos of the insulin blocking effect of seroquel but i do feel sugar crashes when eat fast carbs.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ade_tweet_binary_pipeline", "en", "clinical/models")

val text = "I used to be on paxil but that made me more depressed and prozac made me angry. Maybe cos of the insulin blocking effect of seroquel but i do feel sugar crashes when eat fast carbs."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk        |   begin |   end | ner_label   |   confidence |
|---:|:-----------------|--------:|------:|:------------|-------------:|
|  0 | depressed        |      44 |    52 | ADE         |     0.999755 |
|  1 | angry            |      73 |    77 | ADE         |     0.999608 |
|  2 | insulin blocking |      97 |   112 | ADE         |     0.738712 |
|  3 | sugar crashes    |     147 |   159 | ADE         |     0.993742 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ade_tweet_binary_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel