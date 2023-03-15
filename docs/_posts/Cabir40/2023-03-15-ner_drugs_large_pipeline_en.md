---
layout: model
title: Pipeline to Detect Drug Chemicals
author: John Snow Labs
name: ner_drugs_large_pipeline
date: 2023-03-15
tags: [ner, clinical, licensed, en]
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

This pretrained pipeline is built on the top of [ner_drugs_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_large_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_drugs_large_pipeline_en_4.3.0_3.2_1678867384370.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_drugs_large_pipeline_en_4.3.0_3.2_1678867384370.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_drugs_large_pipeline", "en", "clinical/models")

text = '''The patient is a 40-year-old white male who presents with a chief complaint of 'chest pain'. The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. He has been advised Aspirin 81 milligrams QDay. Humulin N. insulin 50 units in a.m. HCTZ 50 mg QDay. Nitroglycerin 1/150 sublingually PRN chest pain..'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_drugs_large_pipeline", "en", "clinical/models")

val text = "The patient is a 40-year-old white male who presents with a chief complaint of 'chest pain'. The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. He has been advised Aspirin 81 milligrams QDay. Humulin N. insulin 50 units in a.m. HCTZ 50 mg QDay. Nitroglycerin 1/150 sublingually PRN chest pain.."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                        |   begin |   end | ner_label   |   confidence |
|---:|:---------------------------------|--------:|------:|:------------|-------------:|
|  0 | Aspirin 81 milligrams            |     306 |   326 | DRUG        |     0.8401   |
|  1 | Humulin N                        |     334 |   342 | DRUG        |     0.95755  |
|  2 | insulin 50 units                 |     345 |   360 | DRUG        |     0.847067 |
|  3 | HCTZ 50 mg                       |     370 |   379 | DRUG        |     0.875567 |
|  4 | Nitroglycerin 1/150 sublingually |     387 |   418 | DRUG        |     0.845967 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_drugs_large_pipeline|
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