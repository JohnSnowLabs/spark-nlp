---
layout: model
title: Pipeline to Detect Clinical Conditions (ner_eu_clinical_condition - es)
author: John Snow Labs
name: ner_eu_clinical_condition_pipeline
date: 2023-03-08
tags: [es, clinical, licensed, ner, clinical_condition]
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

This pretrained pipeline is built on the top of [ner_eu_clinical_condition](https://nlp.johnsnowlabs.com/2023/02/06/ner_eu_clinical_condition_es.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_pipeline_es_4.3.0_3.2_1678260469189.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_pipeline_es_4.3.0_3.2_1678260469189.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_eu_clinical_condition_pipeline", "es", "clinical/models")

text = "
La exploración abdominal revela una cicatriz de laparotomía media infraumbilical, la presencia de ruidos disminuidos, y dolor a la palpación de manera difusa sin claros signos de irritación peritoneal. No existen hernias inguinales o crurales.
"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_eu_clinical_condition_pipeline", "es", "clinical/models")

val text = "
La exploración abdominal revela una cicatriz de laparotomía media infraumbilical, la presencia de ruidos disminuidos, y dolor a la palpación de manera difusa sin claros signos de irritación peritoneal. No existen hernias inguinales o crurales.
"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks               |   begin |   end | entities           |   confidence |
|---:|:---------------------|--------:|------:|:-------------------|-------------:|
|  0 | cicatriz             |      37 |    44 | clinical_condition |      0.9883  |
|  1 | dolor a la palpación |     121 |   140 | clinical_condition |      0.87025 |
|  2 | signos               |     170 |   175 | clinical_condition |      0.9862  |
|  3 | irritación           |     180 |   189 | clinical_condition |      0.9975  |
|  4 | hernias inguinales   |     214 |   231 | clinical_condition |      0.7543  |
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
|Language:|es|
|Size:|1.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel