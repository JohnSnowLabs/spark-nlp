---
layout: model
title: Pipeline to Detect Pathogen, Medical Condition and Medicine (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_pathogen_pipeline
date: 2023-03-20
tags: [licensed, clinical, en, ner, pathogen, medical_condition, medicine, berfortokenclassification]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_pathogen](https://nlp.johnsnowlabs.com/2022/07/28/bert_token_classifier_ner_pathogen_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_pathogen_pipeline_en_4.3.0_3.2_1679299357172.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_pathogen_pipeline_en_4.3.0_3.2_1679299357172.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_pathogen_pipeline", "en", "clinical/models")

text = '''Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin. This can progress to loss of skin color, a fast heart rate as it becomes more severe; while it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_pathogen_pipeline", "en", "clinical/models")

val text = "Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin. This can progress to loss of skin color, a fast heart rate as it becomes more severe; while it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk       |   begin |   end | ner_label        |   confidence |
|---:|:----------------|--------:|------:|:-----------------|-------------:|
|  0 | Racecadotril    |       0 |    11 | Medicine         |     0.986453 |
|  1 | loperamide      |      80 |    89 | Medicine         |     0.967653 |
|  2 | Diarrhea        |      92 |    99 | MedicalCondition |     0.92107  |
|  3 | loose           |     128 |   132 | MedicalCondition |     0.639717 |
|  4 | liquid          |     135 |   140 | MedicalCondition |     0.739769 |
|  5 | watery          |     145 |   150 | MedicalCondition |     0.911771 |
|  6 | bowel movements |     152 |   166 | MedicalCondition |     0.637392 |
|  7 | dehydration     |     187 |   197 | MedicalCondition |     0.81079  |
|  8 | loss            |     282 |   285 | MedicalCondition |     0.526605 |
|  9 | color           |     295 |   299 | MedicalCondition |     0.612506 |
| 10 | fast            |     304 |   307 | MedicalCondition |     0.555894 |
| 11 | heart rate      |     309 |   318 | MedicalCondition |     0.486794 |
| 12 | rabies virus    |     381 |   392 | Pathogen         |     0.738198 |
| 13 | Lyssavirus      |     395 |   404 | Pathogen         |     0.979239 |
| 14 | Ephemerovirus   |     410 |   422 | Pathogen         |     0.992292 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_pathogen_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel