---
layout: model
title: Pipeline to Detect Clinical Entities (ner_eu_clinical_case)
author: John Snow Labs
name: ner_eu_clinical_case_pipeline
date: 2023-03-08
tags: [clinical, licensed, ner, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_eu_clinical_case](https://nlp.johnsnowlabs.com/2023/01/25/ner_eu_clinical_case_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_pipeline_en_4.3.0_3.2_1678262043022.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_pipeline_en_4.3.0_3.2_1678262043022.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_eu_clinical_case_pipeline", "en", "clinical/models")

text = "
A 3-year-old boy with autistic disorder on hospital of pediatric ward A at university hospital. He has no family history of illness or autistic spectrum disorder. The child was diagnosed with a severe communication disorder, with social interaction difficulties and sensory processing delay. Blood work was normal (thyroid-stimulating hormone (TSH), hemoglobin, mean corpuscular volume (MCV), and ferritin). Upper endoscopy also showed a submucosal tumor causing subtotal obstruction of the gastric outlet. Because a gastrointestinal stromal tumor was suspected, distal gastrectomy was performed. Histopathological examination revealed spindle cell proliferation in the submucosal layer.
"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_eu_clinical_case_pipeline", "en", "clinical/models")

val text = "
A 3-year-old boy with autistic disorder on hospital of pediatric ward A at university hospital. He has no family history of illness or autistic spectrum disorder. The child was diagnosed with a severe communication disorder, with social interaction difficulties and sensory processing delay. Blood work was normal (thyroid-stimulating hormone (TSH), hemoglobin, mean corpuscular volume (MCV), and ferritin). Upper endoscopy also showed a submucosal tumor causing subtotal obstruction of the gastric outlet. Because a gastrointestinal stromal tumor was suspected, distal gastrectomy was performed. Histopathological examination revealed spindle cell proliferation in the submucosal layer.
"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks                         |   begin |   end | entities           |   confidence |
|---:|:-------------------------------|--------:|------:|:-------------------|-------------:|
|  0 | A 3-year-old boy               |       1 |    16 | patient            |     0.733133 |
|  1 | autistic disorder              |      23 |    39 | clinical_condition |     0.5412   |
|  2 | He                             |      97 |    98 | patient            |     0.9991   |
|  3 | illness                        |     125 |   131 | clinical_event     |     0.4956   |
|  4 | autistic spectrum disorder     |     136 |   161 | clinical_condition |     0.5002   |
|  5 | The child                      |     164 |   172 | patient            |     0.82435  |
|  6 | diagnosed                      |     178 |   186 | clinical_event     |     0.9912   |
|  7 | disorder                       |     216 |   223 | clinical_event     |     0.3804   |
|  8 | difficulties                   |     250 |   261 | clinical_event     |     0.3221   |
|  9 | Blood                          |     293 |   297 | bodypart           |     0.7617   |
| 10 | work                           |     299 |   302 | clinical_event     |     0.9361   |
| 11 | normal                         |     308 |   313 | units_measurements |     0.5337   |
| 12 | hormone                        |     336 |   342 | clinical_event     |     0.362    |
| 13 | hemoglobin                     |     351 |   360 | clinical_event     |     0.6106   |
| 14 | volume                         |     380 |   385 | clinical_event     |     0.6226   |
| 15 | endoscopy                      |     415 |   423 | clinical_event     |     0.9917   |
| 16 | showed                         |     430 |   435 | clinical_event     |     0.9904   |
| 17 | tumor                          |     450 |   454 | clinical_condition |     0.5606   |
| 18 | causing                        |     456 |   462 | clinical_event     |     0.7362   |
| 19 | obstruction                    |     473 |   483 | clinical_event     |     0.6198   |
| 20 | the gastric outlet             |     488 |   505 | bodypart           |     0.634967 |
| 21 | gastrointestinal stromal tumor |     518 |   547 | clinical_condition |     0.387833 |
| 22 | suspected                      |     553 |   561 | clinical_event     |     0.8225   |
| 23 | gastrectomy                    |     571 |   581 | clinical_event     |     0.935    |
| 24 | examination                    |     616 |   626 | clinical_event     |     0.9987   |
| 25 | revealed                       |     628 |   635 | clinical_event     |     0.9989   |
| 26 | spindle cell proliferation     |     637 |   662 | clinical_condition |     0.4487   |
| 27 | the submucosal layer           |     667 |   686 | bodypart           |     0.523    |
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
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel