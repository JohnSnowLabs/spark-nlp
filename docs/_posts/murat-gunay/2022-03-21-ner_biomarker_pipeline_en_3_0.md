---
layout: model
title: Pipeline to Extraction of biomarker information
author: John Snow Labs
name: ner_biomarker_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_biomarker](https://nlp.johnsnowlabs.com/2021/11/26/ner_biomarker_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_biomarker_pipeline_en_3.4.1_3.0_1647871954538.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_biomarker_pipeline_en_3.4.1_3.0_1647871954538.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_biomarker_pipeline", "en", "clinical/models")


pipeline.annotate("Here , we report the first case of an intraductal tubulopapillary neoplasm of the pancreas with clear cell morphology . Immunohistochemistry revealed positivity for Pan-CK , CK7 , CK8/18 , MUC1 , MUC6 , carbonic anhydrase IX , CD10 , EMA , β-catenin and e-cadherin ")
```
```scala
val pipeline = new PretrainedPipeline("ner_biomarker_pipeline", "en", "clinical/models")


pipeline.annotate("Here , we report the first case of an intraductal tubulopapillary neoplasm of the pancreas with clear cell morphology . Immunohistochemistry revealed positivity for Pan-CK , CK7 , CK8/18 , MUC1 , MUC6 , carbonic anhydrase IX , CD10 , EMA , β-catenin and e-cadherin ")
```
</div>

## Results

```bash
|    | ner_chunk                | entity                |   confidence |
|---:|:-------------------------|:----------------------|-------------:|
|  0 | intraductal              | CancerModifier        |     0.9934   |
|  1 | tubulopapillary          | CancerModifier        |     0.6403   |
|  2 | neoplasm of the pancreas | CancerDx              |     0.758825 |
|  3 | clear cell               | CancerModifier        |     0.9633   |
|  4 | Immunohistochemistry     | Test                  |     0.9534   |
|  5 | positivity               | Biomarker_Measurement |     0.8795   |
|  6 | Pan-CK                   | Biomarker             |     0.9975   |
|  7 | CK7                      | Biomarker             |     0.9975   |
|  8 | CK8/18                   | Biomarker             |     0.9987   |
|  9 | MUC1                     | Biomarker             |     0.9967   |
| 10 | MUC6                     | Biomarker             |     0.9972   |
| 11 | carbonic anhydrase IX    | Biomarker             |     0.937567 |
| 12 | CD10                     | Biomarker             |     0.9974   |
| 13 | EMA                      | Biomarker             |     0.9899   |
| 14 | β-catenin                | Biomarker             |     0.8059   |
| 15 | e-cadherin               | Biomarker             |     0.9806   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_biomarker_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
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
- NerConverter