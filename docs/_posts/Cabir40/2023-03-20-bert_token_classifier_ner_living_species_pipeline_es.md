---
layout: model
title: Pipeline to Detect Living Species
author: John Snow Labs
name: bert_token_classifier_ner_living_species_pipeline
date: 2023-03-20
tags: [es, ner, clinical, licensed, bertfortokenclassification]
task: Named Entity Recognition
language: es
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_token_classifier_ner_living_species](https://nlp.johnsnowlabs.com/2022/06/27/bert_token_classifier_ner_living_species_es_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_living_species_pipeline_es_4.3.0_3.2_1679304476657.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_living_species_pipeline_es_4.3.0_3.2_1679304476657.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_living_species_pipeline", "es", "clinical/models")

text = '''Lactante varón de dos años. Antecedentes familiares sin interés. Antecedentes personales: Embarazo, parto y periodo neonatal normal. En seguimiento por alergia a legumbres, diagnosticado con diez meses por reacción urticarial generalizada con lentejas y garbanzos, con dieta de exclusión a legumbres desde entonces. En ésta visita la madre describe episodios de eritema en zona maxilar derecha con afectación ocular ipsilateral que se resuelve en horas tras la administración de corticoides. Le ha ocurrido en 5-6 ocasiones, en relación con la ingesta de alimentos previamente tolerados. Exploración complementaria: Cacahuete, ac(ige)19.2 Ku.arb/l. Resultados: Ante la sospecha clínica de Síndrome de Frey, se tranquiliza a los padres, explicándoles la naturaleza del cuadro y se cita para revisión anual.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_living_species_pipeline", "es", "clinical/models")

val text = "Lactante varón de dos años. Antecedentes familiares sin interés. Antecedentes personales: Embarazo, parto y periodo neonatal normal. En seguimiento por alergia a legumbres, diagnosticado con diez meses por reacción urticarial generalizada con lentejas y garbanzos, con dieta de exclusión a legumbres desde entonces. En ésta visita la madre describe episodios de eritema en zona maxilar derecha con afectación ocular ipsilateral que se resuelve en horas tras la administración de corticoides. Le ha ocurrido en 5-6 ocasiones, en relación con la ingesta de alimentos previamente tolerados. Exploración complementaria: Cacahuete, ac(ige)19.2 Ku.arb/l. Resultados: Ante la sospecha clínica de Síndrome de Frey, se tranquiliza a los padres, explicándoles la naturaleza del cuadro y se cita para revisión anual."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk      |   begin |   end | ner_label   |   confidence |
|---:|:---------------|--------:|------:|:------------|-------------:|
|  0 | Lactante varón |       0 |    13 | HUMAN       |     0.999294 |
|  1 | familiares     |      41 |    50 | HUMAN       |     0.999974 |
|  2 | personales     |      78 |    87 | HUMAN       |     0.999983 |
|  3 | neonatal       |     116 |   123 | HUMAN       |     0.999961 |
|  4 | legumbres      |     162 |   170 | SPECIES     |     0.999973 |
|  5 | lentejas       |     243 |   250 | SPECIES     |     0.999977 |
|  6 | garbanzos      |     254 |   262 | SPECIES     |     0.99997  |
|  7 | legumbres      |     290 |   298 | SPECIES     |     0.999974 |
|  8 | madre          |     334 |   338 | HUMAN       |     0.999971 |
|  9 | Cacahuete      |     616 |   624 | SPECIES     |     0.99997  |
| 10 | padres         |     728 |   733 | HUMAN       |     0.999971 |
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
|Language:|es|
|Size:|410.6 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel