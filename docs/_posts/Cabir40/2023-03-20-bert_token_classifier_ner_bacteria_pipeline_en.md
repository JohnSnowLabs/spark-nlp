---
layout: model
title: Pipeline to Detect Bacterial Species (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_bacteria_pipeline
date: 2023-03-20
tags: [bacteria, bertfortokenclassification, ner, en, licensed]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_bacteria](https://nlp.johnsnowlabs.com/2022/01/07/bert_token_classifier_ner_bacteria_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bacteria_pipeline_en_4.3.0_3.2_1679305685030.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bacteria_pipeline_en_4.3.0_3.2_1679305685030.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_bacteria_pipeline", "en", "clinical/models")

text = '''Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T)).'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_bacteria_pipeline", "en", "clinical/models")

val text = "Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T))."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk               |   begin |   end | ner_label   |   confidence |
|---:|:------------------------|--------:|------:|:------------|-------------:|
|  0 | SMSP (T)                |      73 |    80 | SPECIES     |     0.99985  |
|  1 | Methanoregula formicica |     167 |   189 | SPECIES     |     0.999787 |
|  2 | SMSP (T)                |     222 |   229 | SPECIES     |     0.999871 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bacteria_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel