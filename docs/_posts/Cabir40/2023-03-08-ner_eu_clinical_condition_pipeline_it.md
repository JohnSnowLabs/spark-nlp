---
layout: model
title: Pipeline to Detect Clinical Conditions (ner_eu_clinical_condition - it)
author: John Snow Labs
name: ner_eu_clinical_condition_pipeline
date: 2023-03-08
tags: [it, clinical, licensed, ner, clinical_condition]
task: Named Entity Recognition
language: it
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_eu_clinical_condition](https://nlp.johnsnowlabs.com/2023/02/06/ner_eu_clinical_condition_it.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_pipeline_it_4.3.0_3.2_1678258845491.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_pipeline_it_4.3.0_3.2_1678258845491.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_eu_clinical_condition_pipeline", "it", "clinical/models")

text = "
Donna, 64 anni, ricovero per dolore epigastrico persistente, irradiato a barra e posteriormente, associato a dispesia e anoressia. Poche settimane dopo compaiono, però, iperemia, intenso edema vulvare ed una esione ulcerativa sul lato sinistro della parete rettale che la RM mostra essere una fistola transfinterica. Questi trattamenti determinano un miglioramento dell’ infiammazione ed una riduzione dell’ ulcera, ma i condilomi permangono inalterati.

"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_eu_clinical_condition_pipeline", "it", "clinical/models")

val text = "
Donna, 64 anni, ricovero per dolore epigastrico persistente, irradiato a barra e posteriormente, associato a dispesia e anoressia. Poche settimane dopo compaiono, però, iperemia, intenso edema vulvare ed una esione ulcerativa sul lato sinistro della parete rettale che la RM mostra essere una fistola transfinterica. Questi trattamenti determinano un miglioramento dell’ infiammazione ed una riduzione dell’ ulcera, ma i condilomi permangono inalterati.

"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks                 |   begin |   end | entities           |   confidence |
|---:|:-----------------------|--------:|------:|:-------------------|-------------:|
|  0 | dolore epigastrico     |      30 |    47 | clinical_condition |      0.90845 |
|  1 | anoressia              |     121 |   129 | clinical_condition |      0.9998  |
|  2 | iperemia               |     170 |   177 | clinical_condition |      0.9999  |
|  3 | edema                  |     188 |   192 | clinical_condition |      1       |
|  4 | fistola transfinterica |     294 |   315 | clinical_condition |      0.97785 |
|  5 | infiammazione          |     372 |   384 | clinical_condition |      0.9996  |
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
|Language:|it|
|Size:|1.2 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel