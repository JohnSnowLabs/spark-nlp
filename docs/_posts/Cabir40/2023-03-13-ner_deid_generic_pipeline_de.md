---
layout: model
title: Pipeline to Detect PHI for Deidentification (Generic)
author: John Snow Labs
name: ner_deid_generic_pipeline
date: 2023-03-13
tags: [deid, ner, de, licensed]
task: Named Entity Recognition
language: de
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_deid_generic](https://nlp.johnsnowlabs.com/2022/01/06/ner_deid_generic_de.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_pipeline_de_4.3.0_3.2_1678743247498.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_pipeline_de_4.3.0_3.2_1678743247498.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_deid_generic_pipeline", "de", "clinical/models")

text = '''Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhaus in Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_deid_generic_pipeline", "de", "clinical/models")

val text = "Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhaus in Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                |   begin |   end | ner_label   |   confidence |
|---:|:--------------------------|--------:|------:|:------------|-------------:|
|  0 | Michael Berger            |       0 |    13 | NAME        |     0.99555  |
|  1 | 12 Dezember 2018          |      34 |    49 | DATE        |     0.999967 |
|  2 | St. Elisabeth-Krankenhaus |      55 |    79 | LOCATION    |     0.797267 |
|  3 | Bad Kissingen             |      84 |    96 | LOCATION    |     0.90785  |
|  4 | Berger                    |     117 |   122 | NAME        |     0.935    |
|  5 | 76                        |     128 |   129 | AGE         |     1        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_generic_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|de|
|Size:|1.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel