---
layout: model
title: Pipeline to Detect Clinical Conditions (ner_eu_clinical_case - eu)
author: John Snow Labs
name: ner_eu_clinical_condition_pipeline
date: 2023-03-07
tags: [eu, clinical, licensed, ner, clinical_condition]
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

This pretrained pipeline is built on the top of [ner_eu_clinical_condition](https://nlp.johnsnowlabs.com/2023/02/06/ner_eu_clinical_condition_eu.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_pipeline_eu_4.3.0_3.2_1678213509285.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_pipeline_eu_4.3.0_3.2_1678213509285.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_eu_clinical_condition_pipeline", "eu", "clinical/models")

text = "
Gertaera honetatik bi hilabetetara, umea Larrialdietako Zerbitzura dator 4 egunetan zehar buruko mina eta bekokiko hantura azaltzeagatik, sukarrik izan gabe. Miaketan, haztapen mingarria duen bekokiko  hantura bigunaz gain, ez da beste zeinurik azaltzen. Polakiuria eta tenesmo arina ere izan zuen egun horretan hematuriarekin batera. Geroztik sintomarik gabe dago.
"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_eu_clinical_condition_pipeline", "eu", "clinical/models")

val text = "
Gertaera honetatik bi hilabetetara, umea Larrialdietako Zerbitzura dator 4 egunetan zehar buruko mina eta bekokiko hantura azaltzeagatik, sukarrik izan gabe. Miaketan, haztapen mingarria duen bekokiko  hantura bigunaz gain, ez da beste zeinurik azaltzen. Polakiuria eta tenesmo arina ere izan zuen egun horretan hematuriarekin batera. Geroztik sintomarik gabe dago.
"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks     |   begin |   end | entities           |   confidence |
|---:|:-----------|--------:|------:|:-------------------|-------------:|
|  0 | mina       |      98 |   101 | clinical_condition |       0.8754 |
|  1 | hantura    |     116 |   122 | clinical_condition |       0.8877 |
|  2 | sukarrik   |     139 |   146 | clinical_condition |       0.9119 |
|  3 | mingarria  |     178 |   186 | clinical_condition |       0.7381 |
|  4 | hantura    |     203 |   209 | clinical_condition |       0.8805 |
|  5 | Polakiuria |     256 |   265 | clinical_condition |       0.6683 |
|  6 | sintomarik |     345 |   354 | clinical_condition |       0.9632 |
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
|Language:|eu|
|Size:|1.1 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel