---
layout: model
title: Pipeline to Detect Clinical Conditions (ner_eu_clinical_case - fr)
author: John Snow Labs
name: ner_eu_clinical_condition_pipeline
date: 2023-03-08
tags: [fr, clinical, licensed, ner, clinical_condition]
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

This pretrained pipeline is built on the top of [ner_eu_clinical_condition](https://nlp.johnsnowlabs.com/2023/02/06/ner_eu_clinical_condition_fr.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_pipeline_fr_4.3.0_3.2_1678260057351.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_pipeline_fr_4.3.0_3.2_1678260057351.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_eu_clinical_condition_pipeline", "fr", "clinical/models")

text = "
Il aurait présenté il y’ a environ 30 ans des ulcérations génitales non traitées spontanément guéries. L’interrogatoire retrouvait une toux sèche depuis trois mois, des douleurs rétro-sternales constrictives, une dyspnée stade III de la NYHA et un contexte d’ apyrexie. Sur ce tableau s’ est greffé des œdèmes des membres inférieurs puis un tableau d’ anasarque d’ où son hospitalisation en cardiologie pour décompensation cardiaque globale.

"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_eu_clinical_condition_pipeline", "fr", "clinical/models")

val text = "
Il aurait présenté il y’ a environ 30 ans des ulcérations génitales non traitées spontanément guéries. L’interrogatoire retrouvait une toux sèche depuis trois mois, des douleurs rétro-sternales constrictives, une dyspnée stade III de la NYHA et un contexte d’ apyrexie. Sur ce tableau s’ est greffé des œdèmes des membres inférieurs puis un tableau d’ anasarque d’ où son hospitalisation en cardiologie pour décompensation cardiaque globale.

"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks                   |   begin |   end | entities           |   confidence |
|---:|:-------------------------|--------:|------:|:-------------------|-------------:|
|  0 | ulcérations              |      47 |    57 | clinical_condition |       0.9995 |
|  1 | toux sèche               |     136 |   145 | clinical_condition |       0.917  |
|  2 | douleurs                 |     170 |   177 | clinical_condition |       0.9999 |
|  3 | dyspnée                  |     214 |   220 | clinical_condition |       0.9999 |
|  4 | apyrexie                 |     261 |   268 | clinical_condition |       0.9963 |
|  5 | anasarque                |     353 |   361 | clinical_condition |       0.9973 |
|  6 | décompensation cardiaque |     409 |   432 | clinical_condition |       0.8948 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_eu_clinical_condition_pipeline|
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