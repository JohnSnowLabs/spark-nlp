---
layout: model
title: Pipeline to Detect Organism in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_species_pipeline
date: 2023-03-20
tags: [en, ner, clinical, licensed, bertfortokenclassification]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_species](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_species_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_species_pipeline_en_4.3.0_3.2_1679301125473.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_species_pipeline_en_4.3.0_3.2_1679301125473.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_species_pipeline", "en", "clinical/models")

text = '''As determined by 16S rRNA gene sequence analysis, strain 6C (T) represents a distinct species belonging to the class Betaproteobacteria and is most closely related to Thiomonas intermedia DSM 18155 (T) and Thiomonas perometabolis DSM 18570 (T) .'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_species_pipeline", "en", "clinical/models")

val text = "As determined by 16S rRNA gene sequence analysis, strain 6C (T) represents a distinct species belonging to the class Betaproteobacteria and is most closely related to Thiomonas intermedia DSM 18155 (T) and Thiomonas perometabolis DSM 18570 (T) ."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk               |   begin |   end | ner_label   |   confidence |
|---:|:------------------------|--------:|------:|:------------|-------------:|
|  0 | 6C (T)                  |      57 |    62 | SPECIES     |     0.998955 |
|  1 | Betaproteobacteria      |     117 |   134 | SPECIES     |     0.99973  |
|  2 | Thiomonas intermedia    |     167 |   186 | SPECIES     |     0.999822 |
|  3 | DSM 18155 (T)           |     188 |   200 | SPECIES     |     0.997657 |
|  4 | Thiomonas perometabolis |     206 |   228 | SPECIES     |     0.999614 |
|  5 | DSM 18570 (T)           |     230 |   242 | SPECIES     |     0.997146 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_species_pipeline|
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