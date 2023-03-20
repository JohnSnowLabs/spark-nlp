---
layout: model
title: Pipeline to Detect Living Species
author: John Snow Labs
name: bert_token_classifier_ner_living_species_pipeline
date: 2023-03-20
tags: [pt, ner, clinical, licensed, bertfortokenclassification]
task: Named Entity Recognition
language: pt
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_token_classifier_ner_living_species](https://nlp.johnsnowlabs.com/2022/06/27/bert_token_classifier_ner_living_species_pt_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_living_species_pipeline_pt_4.3.0_3.2_1679304320046.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_living_species_pipeline_pt_4.3.0_3.2_1679304320046.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_living_species_pipeline", "pt", "clinical/models")

text = '''Uma rapariga de 16 anos com um historial pessoal de asma apresentou ao departamento de dermatologia com lesões cutâneas assintomáticas que tinham estado presentes durante 2 meses. A paciente tinha sido tratada com creme corticosteróide devido a uma suspeita inicial de eczema atópico, apesar do qual apresentava um crescimento progressivo marcado das lesões. Tinha um gato doméstico que ela nunca tinha levado ao veterinário. O exame físico revelou placas em forma de anel com uma borda periférica activa na parte superior das costas e nos aspectos laterais do pescoço e da face. Cultura local obtida por raspagem de tapete isolado Trichophyton rubrum. Com base em dados clínicos e cultura, foi estabelecido o diagnóstico de tinea incognito.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_living_species_pipeline", "pt", "clinical/models")

val text = "Uma rapariga de 16 anos com um historial pessoal de asma apresentou ao departamento de dermatologia com lesões cutâneas assintomáticas que tinham estado presentes durante 2 meses. A paciente tinha sido tratada com creme corticosteróide devido a uma suspeita inicial de eczema atópico, apesar do qual apresentava um crescimento progressivo marcado das lesões. Tinha um gato doméstico que ela nunca tinha levado ao veterinário. O exame físico revelou placas em forma de anel com uma borda periférica activa na parte superior das costas e nos aspectos laterais do pescoço e da face. Cultura local obtida por raspagem de tapete isolado Trichophyton rubrum. Com base em dados clínicos e cultura, foi estabelecido o diagnóstico de tinea incognito."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk           |   begin |   end | ner_label   |   confidence |
|---:|:--------------------|--------:|------:|:------------|-------------:|
|  0 | rapariga            |       4 |    11 | HUMAN       |     0.999888 |
|  1 | pessoal             |      41 |    47 | HUMAN       |     0.99987  |
|  2 | paciente            |     182 |   189 | HUMAN       |     0.999731 |
|  3 | gato                |     368 |   371 | SPECIES     |     0.999365 |
|  4 | veterinário         |     413 |   423 | HUMAN       |     0.982236 |
|  5 | Trichophyton rubrum |     632 |   650 | SPECIES     |     0.996602 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_living_species_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|pt|
|Size:|666.0 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel