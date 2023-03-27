---
layout: model
title: Pipeline to Detect Living Species(w2v_cc_300d)
author: John Snow Labs
name: ner_living_species_pipeline
date: 2023-03-13
tags: [pt, ner, clinical, licensed]
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

This pretrained pipeline is built on the top of [ner_living_species](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_pt_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_pipeline_pt_4.3.0_3.2_1678708110628.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_living_species_pipeline_pt_4.3.0_3.2_1678708110628.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_living_species_pipeline", "pt", "clinical/models")

text = '''Uma rapariga de 16 anos com um historial pessoal de asma apresentou ao departamento de dermatologia com lesões cutâneas assintomáticas que tinham estado presentes durante 2 meses. A paciente tinha sido tratada com creme corticosteróide devido a uma suspeita inicial de eczema atópico, apesar do qual apresentava um crescimento progressivo marcado das lesões. Tinha um gato doméstico que ela nunca tinha levado ao veterinário. O exame físico revelou placas em forma de anel com uma borda periférica activa na parte superior das costas e nos aspectos laterais do pescoço e da face. Cultura local obtida por raspagem de tapete isolado Trichophyton rubrum. Com base em dados clínicos e cultura, foi estabelecido o diagnóstico de tinea incognito.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_living_species_pipeline", "pt", "clinical/models")

val text = "Uma rapariga de 16 anos com um historial pessoal de asma apresentou ao departamento de dermatologia com lesões cutâneas assintomáticas que tinham estado presentes durante 2 meses. A paciente tinha sido tratada com creme corticosteróide devido a uma suspeita inicial de eczema atópico, apesar do qual apresentava um crescimento progressivo marcado das lesões. Tinha um gato doméstico que ela nunca tinha levado ao veterinário. O exame físico revelou placas em forma de anel com uma borda periférica activa na parte superior das costas e nos aspectos laterais do pescoço e da face. Cultura local obtida por raspagem de tapete isolado Trichophyton rubrum. Com base em dados clínicos e cultura, foi estabelecido o diagnóstico de tinea incognito."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks          |   begin |   end | ner_label   |   confidence |
|---:|:--------------------|--------:|------:|:------------|-------------:|
|  0 | rapariga            |       4 |    11 | HUMAN       |       0.9991 |
|  1 | pessoal             |      41 |    47 | HUMAN       |       0.9765 |
|  2 | paciente            |     182 |   189 | HUMAN       |       1      |
|  3 | gato                |     368 |   371 | SPECIES     |       0.9847 |
|  4 | veterinário         |     413 |   423 | HUMAN       |       0.91   |
|  5 | Trichophyton rubrum |     632 |   650 | SPECIES     |       0.9996 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_living_species_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|pt|
|Size:|1.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel