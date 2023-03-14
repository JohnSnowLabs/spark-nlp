---
layout: model
title: Pipeline to Detect Clinical Entities (ner_eu_clinical_case - eu)
author: John Snow Labs
name: ner_eu_clinical_case_pipeline
date: 2023-03-08
tags: [eu, clinical, licensed, ner]
task: Named Entity Recognition
language: eu
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_eu_clinical_case](https://nlp.johnsnowlabs.com/2023/02/02/ner_eu_clinical_case_eu.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_pipeline_eu_4.3.0_3.2_1678261023976.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_pipeline_eu_4.3.0_3.2_1678261023976.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_eu_clinical_case_pipeline", "eu", "clinical/models")

text = "
3 urteko mutiko bat nahasmendu autistarekin unibertsitateko ospitaleko A pediatriako ospitalean. Ez du autismoaren espektroaren nahaste edo gaixotasun familiaren aurrekaririk. Mutilari komunikazio-nahaste larria diagnostikatu zioten, elkarrekintza sozialeko zailtasunak eta prozesamendu sentsorial atzeratua. Odol-analisiak normalak izan ziren (tiroidearen hormona estimulatzailea (TSH), hemoglobina, batez besteko bolumen corpuskularra (MCV) eta ferritina). Goiko endoskopiak mukosaren azpiko tumore bat ere erakutsi zuen, urdail-irteeren guztizko oztopoa eragiten zuena. Estroma gastrointestinalaren tumore baten susmoa ikusita, distaleko gastrektomia egin zen. Azterketa histopatologikoak agerian utzi zuen mukosaren azpiko zelulen ugaltzea.
"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_eu_clinical_case_pipeline", "eu", "clinical/models")

val text = "
3 urteko mutiko bat nahasmendu autistarekin unibertsitateko ospitaleko A pediatriako ospitalean. Ez du autismoaren espektroaren nahaste edo gaixotasun familiaren aurrekaririk. Mutilari komunikazio-nahaste larria diagnostikatu zioten, elkarrekintza sozialeko zailtasunak eta prozesamendu sentsorial atzeratua. Odol-analisiak normalak izan ziren (tiroidearen hormona estimulatzailea (TSH), hemoglobina, batez besteko bolumen corpuskularra (MCV) eta ferritina). Goiko endoskopiak mukosaren azpiko tumore bat ere erakutsi zuen, urdail-irteeren guztizko oztopoa eragiten zuena. Estroma gastrointestinalaren tumore baten susmoa ikusita, distaleko gastrektomia egin zen. Azterketa histopatologikoak agerian utzi zuen mukosaren azpiko zelulen ugaltzea.
"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks                   |   begin |   end | entities           |   confidence |
|---:|:-------------------------|--------:|------:|:-------------------|-------------:|
|  0 | 3 urteko mutiko bat      |       1 |    19 | patient            |     0.813975 |
|  1 | nahasmendu               |      21 |    30 | clinical_event     |     0.9848   |
|  2 | autismoaren espektroaren |     104 |   127 | clinical_condition |     0.344    |
|  3 | nahaste                  |     129 |   135 | clinical_event     |     0.996    |
|  4 | gaixotasun               |     141 |   150 | clinical_event     |     0.9839   |
|  5 | familiaren               |     152 |   161 | patient            |     0.8834   |
|  6 | aurrekaririk             |     163 |   174 | clinical_event     |     0.8742   |
|  7 | Mutilari                 |     177 |   184 | patient            |     0.9477   |
|  8 | komunikazio-nahaste      |     186 |   204 | clinical_event     |     0.8647   |
|  9 | diagnostikatu            |     213 |   225 | clinical_event     |     0.9969   |
| 10 | elkarrekintza            |     235 |   247 | clinical_event     |     0.9828   |
| 11 | zailtasunak              |     259 |   269 | clinical_event     |     0.9897   |
| 12 | prozesamendu             |     275 |   286 | clinical_event     |     0.9927   |
| 13 | sentsorial               |     288 |   297 | clinical_condition |     0.7912   |
| 14 | Odol-analisiak           |     310 |   323 | clinical_event     |     0.9992   |
| 15 | normalak                 |     325 |   332 | units_measurements |     0.7265   |
| 16 | tiroidearen              |     346 |   356 | bodypart           |     0.9718   |
| 17 | hormona                  |     358 |   364 | clinical_event     |     0.9904   |
| 18 | estimulatzailea          |     366 |   380 | clinical_condition |     0.6005   |
| 19 | TSH                      |     383 |   385 | clinical_event     |     0.9976   |
| 20 | hemoglobina              |     389 |   399 | clinical_event     |     0.9936   |
| 21 | bolumen                  |     416 |   422 | clinical_event     |     0.735    |
| 22 | MCV                      |     439 |   441 | clinical_event     |     0.9933   |
| 23 | ferritina                |     448 |   456 | clinical_event     |     0.4228   |
| 24 | Goiko                    |     460 |   464 | bodypart           |     0.9564   |
| 25 | endoskopiak              |     466 |   476 | clinical_event     |     0.9082   |
| 26 | mukosaren azpiko         |     478 |   493 | bodypart           |     0.5929   |
| 27 | tumore                   |     495 |   500 | clinical_event     |     0.998    |
| 28 | erakutsi                 |     510 |   517 | clinical_event     |     0.9963   |
| 29 | oztopoa                  |     550 |   556 | clinical_event     |     0.9964   |
| 30 | Estroma                  |     574 |   580 | clinical_event     |     0.884    |
| 31 | gastrointestinalaren     |     582 |   601 | clinical_condition |     0.3525   |
| 32 | tumore                   |     603 |   608 | clinical_event     |     0.9896   |
| 33 | ikusita                  |     623 |   629 | clinical_event     |     0.9873   |
| 34 | distaleko                |     632 |   640 | bodypart           |     0.7425   |
| 35 | gastrektomia             |     642 |   653 | clinical_event     |     0.9986   |
| 36 | Azterketa                |     665 |   673 | clinical_event     |     0.9517   |
| 37 | agerian                  |     693 |   699 | clinical_event     |     0.9842   |
| 38 | utzi                     |     701 |   704 | clinical_event     |     0.925    |
| 39 | mukosaren azpiko zelulen |     711 |   734 | bodypart           |     0.754933 |
| 40 | ugaltzea                 |     736 |   743 | clinical_event     |     0.9989   |
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
|Language:|eu|
|Size:|1.1 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel