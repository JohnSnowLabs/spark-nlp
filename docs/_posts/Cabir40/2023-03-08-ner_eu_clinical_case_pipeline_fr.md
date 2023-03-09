---
layout: model
title: Pipeline to Detect Clinical Entities (ner_eu_clinical_case - fr)
author: John Snow Labs
name: ner_eu_clinical_case_pipeline
date: 2023-03-08
tags: [fr, clinical, licensed, ner]
task: Named Entity Recognition
language: fr
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_eu_clinical_case](https://nlp.johnsnowlabs.com/2023/02/01/ner_eu_clinical_case_fr.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_pipeline_fr_4.3.0_3.2_1678261744783.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_pipeline_fr_4.3.0_3.2_1678261744783.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_eu_clinical_case_pipeline", "fr", "clinical/models")

text = "
Un garçon de 3 ans atteint d'un trouble autistique à l'hôpital du service pédiatrique A de l'hôpital universitaire. Il n'a pas d'antécédents familiaux de troubles ou de maladies du spectre autistique. Le garçon a été diagnostiqué avec un trouble de communication sévère, avec des difficultés d'interaction sociale et un traitement sensoriel retardé. Les tests sanguins étaient normaux (thyréostimuline (TSH), hémoglobine, volume globulaire moyen (MCV) et ferritine). L'endoscopie haute a également montré une tumeur sous-muqueuse provoquant une obstruction subtotale de la sortie gastrique. Devant la suspicion d'une tumeur stromale gastro-intestinale, une gastrectomie distale a été réalisée. L'examen histopathologique a révélé une prolifération de cellules fusiformes dans la couche sous-muqueuse.
"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_eu_clinical_case_pipeline", "fr", "clinical/models")

val text = "
Un garçon de 3 ans atteint d'un trouble autistique à l'hôpital du service pédiatrique A de l'hôpital universitaire. Il n'a pas d'antécédents familiaux de troubles ou de maladies du spectre autistique. Le garçon a été diagnostiqué avec un trouble de communication sévère, avec des difficultés d'interaction sociale et un traitement sensoriel retardé. Les tests sanguins étaient normaux (thyréostimuline (TSH), hémoglobine, volume globulaire moyen (MCV) et ferritine). L'endoscopie haute a également montré une tumeur sous-muqueuse provoquant une obstruction subtotale de la sortie gastrique. Devant la suspicion d'une tumeur stromale gastro-intestinale, une gastrectomie distale a été réalisée. L'examen histopathologique a révélé une prolifération de cellules fusiformes dans la couche sous-muqueuse.
"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks                                                |   begin |   end | entities           |   confidence |
|---:|:------------------------------------------------------|--------:|------:|:-------------------|-------------:|
|  0 | Un garçon de 3 ans                                    |       1 |    18 | patient            |     0.58786  |
|  1 | trouble autistique à l'hôpital du service pédiatrique |      33 |    85 | clinical_condition |     0.560657 |
|  2 | l'hôpital                                             |      92 |   100 | clinical_event     |     0.3725   |
|  3 | Il n'a                                                |     117 |   122 | patient            |     0.62695  |
|  4 | d'antécédents                                         |     128 |   140 | clinical_event     |     0.8355   |
|  5 | troubles                                              |     155 |   162 | clinical_condition |     0.9096   |
|  6 | maladies                                              |     170 |   177 | clinical_condition |     0.9109   |
|  7 | du spectre autistique                                 |     179 |   199 | bodypart           |     0.4828   |
|  8 | Le garçon                                             |     202 |   210 | patient            |     0.48925  |
|  9 | diagnostiqué                                          |     218 |   229 | clinical_event     |     0.2155   |
| 10 | trouble                                               |     239 |   245 | clinical_condition |     0.8545   |
| 11 | difficultés                                           |     281 |   291 | clinical_event     |     0.5636   |
| 12 | traitement                                            |     321 |   330 | clinical_event     |     0.9046   |
| 13 | tests                                                 |     355 |   359 | clinical_event     |     0.9305   |
| 14 | normaux                                               |     378 |   384 | units_measurements |     0.9394   |
| 15 | thyréostimuline                                       |     387 |   401 | clinical_event     |     0.4653   |
| 16 | TSH                                                   |     404 |   406 | clinical_event     |     0.691    |
| 17 | ferritine                                             |     456 |   464 | clinical_event     |     0.2768   |
| 18 | L'endoscopie                                          |     468 |   479 | clinical_event     |     0.7778   |
| 19 | montré                                                |     499 |   504 | clinical_event     |     0.9829   |
| 20 | tumeur sous-muqueuse                                  |     510 |   529 | clinical_condition |     0.7923   |
| 21 | provoquant                                            |     531 |   540 | clinical_event     |     0.868    |
| 22 | obstruction                                           |     546 |   556 | clinical_condition |     0.9448   |
| 23 | la sortie gastrique                                   |     571 |   589 | bodypart           |     0.496233 |
| 24 | suspicion                                             |     602 |   610 | clinical_event     |     0.9035   |
| 25 | tumeur stromale gastro-intestinale                    |     618 |   651 | clinical_condition |     0.5901   |
| 26 | gastrectomie                                          |     658 |   669 | clinical_event     |     0.3939   |
| 27 | L'examen                                              |     695 |   702 | clinical_event     |     0.5114   |
| 28 | révélé                                                |     724 |   729 | clinical_event     |     0.9731   |
| 29 | prolifération                                         |     735 |   747 | clinical_event     |     0.6767   |
| 30 | cellules fusiformes                                   |     752 |   770 | bodypart           |     0.5233   |
| 31 | la couche sous-muqueuse                               |     777 |   799 | bodypart           |     0.6755   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_eu_clinical_case_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|fr|
|Size:|1.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel