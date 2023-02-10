---
layout: model
title: Pipeline to Detect Pathogen, Medical Condition and Medicine
author: John Snow Labs
name: ner_pathogen_pipeline
date: 2022-06-29
tags: [licensed, clinical, en, pathogen, ner, medicine, medical_condition, pipeline]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
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
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_pathogen_pipeline_en_4.0.0_3.0_1656527387514.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_pathogen_pipeline_en_4.0.0_3.0_1656527387514.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_pathogen_pipeline", "en", "clinical/models")

result = pipeline.fullAnnotate("""Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin.  This can progress to loss of skin color, a fast heart rate as it becomes more severe.  While it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions.""")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_pathogen_pipeline", "en", "clinical/models")

val result = pipeline.fullAnnotate("""Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin.  This can progress to loss of skin color, a fast heart rate as it becomes more severe.  While it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions.""")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.pathogen.pipeline").predict("""Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin.  This can progress to loss of skin color, a fast heart rate as it becomes more severe.  While it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions.""")
```

</div>

## Results

```bash
|chunk          |ner_label       |
+---------------+----------------+
|Racecadotril   |Medicine        |
|loperamide     |Medicine        |
|Diarrhea       |MedicalCondition|
|dehydration    |MedicalCondition|
|skin color     |MedicalCondition|
|fast heart rate|MedicalCondition|
|rabies virus   |Pathogen        |
|Lyssavirus     |Pathogen        |
|Ephemerovirus  |Pathogen        |
+---------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_pathogen_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.0.0+|
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
- NerConverter
