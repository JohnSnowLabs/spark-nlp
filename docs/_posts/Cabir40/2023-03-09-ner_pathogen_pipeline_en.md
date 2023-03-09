---
layout: model
title: Pipeline to Detect Pathogen, Medical Condition and Medicine
author: John Snow Labs
name: ner_pathogen_pipeline
date: 2023-03-09
tags: [licensed, clinical, en, ner, pathogen, medical_condition, medicine]
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

This pretrained pipeline is built on the top of [ner_pathogen](https://nlp.johnsnowlabs.com/2022/06/28/ner_pathogen_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_pathogen_pipeline_en_4.3.0_3.2_1678385356869.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_pathogen_pipeline_en_4.3.0_3.2_1678385356869.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_pathogen_pipeline", "en", "clinical/models")

text = '''Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin.  This can progress to loss of skin color, a fast heart rate as it becomes more severe.  While it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_pathogen_pipeline", "en", "clinical/models")

val text = "Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin.  This can progress to loss of skin color, a fast heart rate as it becomes more severe.  While it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks      |   begin |   end | ner_label        |   confidence |
|---:|:----------------|--------:|------:|:-----------------|-------------:|
|  0 | Racecadotril    |       0 |    11 | Medicine         |     0.9468   |
|  1 | loperamide      |      80 |    89 | Medicine         |     0.9986   |
|  2 | Diarrhea        |      92 |    99 | MedicalCondition |     0.9848   |
|  3 | dehydration     |     187 |   197 | MedicalCondition |     0.6305   |
|  4 | skin color      |     291 |   300 | MedicalCondition |     0.6586   |
|  5 | fast heart rate |     305 |   319 | MedicalCondition |     0.757233 |
|  6 | rabies virus    |     383 |   394 | Pathogen         |     0.95685  |
|  7 | Lyssavirus      |     397 |   406 | Pathogen         |     0.9694   |
|  8 | Ephemerovirus   |     412 |   424 | Pathogen         |     0.6919   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_pathogen_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel