---
layout: model
title: Pipeline to Extract Negation and Uncertainty Entities from Spanish Medical Texts (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_negation_uncertainty_pipeline
date: 2023-03-20
tags: [es, clinical, licensed, token_classification, bert, ner, negation, uncertainty, linguistics]
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

This pretrained pipeline is built on the top of [bert_token_classifier_negation_uncertainty](https://nlp.johnsnowlabs.com/2022/08/11/bert_token_classifier_negation_uncertainty_es_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_negation_uncertainty_pipeline_es_4.3.0_3.2_1679298806721.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_negation_uncertainty_pipeline_es_4.3.0_3.2_1679298806721.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_negation_uncertainty_pipeline", "es", "clinical/models")

text = '''Con diagnóstico probable de cirrosis hepática (no conocida previamente) y peritonitis espontanea primaria con tratamiento durante 8 dias con ceftriaxona en el primer ingreso (no se realizó paracentesis control por escasez de liquido). Lesión tumoral en hélix izquierdo de 0,5 cms. de diámetro susceptible de ca basocelular perlado.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_negation_uncertainty_pipeline", "es", "clinical/models")

val text = "Con diagnóstico probable de cirrosis hepática (no conocida previamente) y peritonitis espontanea primaria con tratamiento durante 8 dias con ceftriaxona en el primer ingreso (no se realizó paracentesis control por escasez de liquido). Lesión tumoral en hélix izquierdo de 0,5 cms. de diámetro susceptible de ca basocelular perlado."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                              |   begin |   end | ner_label   |   confidence |
|---:|:-------------------------------------------------------|--------:|------:|:------------|-------------:|
|  0 | probable                                               |      16 |    23 | UNC         |     0.999994 |
|  1 | de cirrosis hepática                                   |      25 |    44 | USCO        |     0.999988 |
|  2 | no                                                     |      47 |    48 | NEG         |     0.999995 |
|  3 | conocida previamente                                   |      50 |    69 | NSCO        |     0.999992 |
|  4 | no                                                     |     175 |   176 | NEG         |     0.999995 |
|  5 | se realizó paracentesis control por escasez de liquido |     178 |   231 | NSCO        |     0.999995 |
|  6 | susceptible de                                         |     293 |   306 | UNC         |     0.999986 |
|  7 | ca basocelular perlado                                 |     308 |   329 | USCO        |     0.99999  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_negation_uncertainty_pipeline|
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