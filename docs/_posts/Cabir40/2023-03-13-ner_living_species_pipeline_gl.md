---
layout: model
title: Pipeline to Detect Living Species (w2v_cc_300d)
author: John Snow Labs
name: ner_living_species_pipeline
date: 2023-03-13
tags: [gl, ner, clinical, licensed]
task: Named Entity Recognition
language: gl
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_living_species](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_gl_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_pipeline_gl_4.3.0_3.2_1678704830024.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_living_species_pipeline_gl_4.3.0_3.2_1678704830024.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_living_species_pipeline", "gl", "clinical/models")

text = '''Muller de 45 anos, sen antecedentes médicos de interese, que foi remitida á consulta de dermatoloxía de urxencias por lesións faciales de tres semanas de evolución. A paciente non presentaba lesións noutras localizaciones nin outra clínica de interese. No seu centro de saúde prescribíronlle corticoides tópicos ante a sospeita de picaduras de artrópodos e unha semana despois, antivirales orais baixo o diagnóstico de posible infección herpética. As lesións interferían de forma notable na súa vida persoal e profesional xa que traballaba de face ao púbico. Unha semana máis tarde o diagnóstico foi confirmado ao resultar o cultivo positivo a Staphylococcus aureus.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_living_species_pipeline", "gl", "clinical/models")

val text = "Muller de 45 anos, sen antecedentes médicos de interese, que foi remitida á consulta de dermatoloxía de urxencias por lesións faciales de tres semanas de evolución. A paciente non presentaba lesións noutras localizaciones nin outra clínica de interese. No seu centro de saúde prescribíronlle corticoides tópicos ante a sospeita de picaduras de artrópodos e unha semana despois, antivirales orais baixo o diagnóstico de posible infección herpética. As lesións interferían de forma notable na súa vida persoal e profesional xa que traballaba de face ao púbico. Unha semana máis tarde o diagnóstico foi confirmado ao resultar o cultivo positivo a Staphylococcus aureus."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks            |   begin |   end | ner_label   |   confidence |
|---:|:----------------------|--------:|------:|:------------|-------------:|
|  0 | Muller                |       0 |     5 | HUMAN       |      0.9998  |
|  1 | paciente              |     167 |   174 | HUMAN       |      0.9985  |
|  2 | artrópodos            |     344 |   353 | SPECIES     |      0.9647  |
|  3 | antivirales           |     378 |   388 | SPECIES     |      0.8854  |
|  4 | herpética             |     437 |   445 | SPECIES     |      0.9592  |
|  5 | púbico                |     551 |   556 | HUMAN       |      0.7293  |
|  6 | Staphylococcus aureus |     644 |   664 | SPECIES     |      0.87005 |
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
|Language:|gl|
|Size:|794.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel