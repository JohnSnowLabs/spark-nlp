---
layout: model
title: Pipeline to Extraction of biomarker information
author: John Snow Labs
name: ner_biomarker_pipeline
date: 2023-03-14
tags: [en, ner, clinical, licensed]
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

This pretrained pipeline is built on the top of [ner_biomarker](https://nlp.johnsnowlabs.com/2021/11/26/ner_biomarker_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_biomarker_pipeline_en_4.3.0_3.2_1678777993811.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_biomarker_pipeline_en_4.3.0_3.2_1678777993811.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_biomarker_pipeline", "en", "clinical/models")

text = '''Here , we report the first case of an intraductal tubulopapillary neoplasm of the pancreas with clear cell morphology . Immunohistochemistry revealed positivity for Pan-CK , CK7 , CK8/18 , MUC1 , MUC6 , carbonic anhydrase IX , CD10 , EMA , β-catenin and e-cadherin '''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_biomarker_pipeline", "en", "clinical/models")

val text = "Here , we report the first case of an intraductal tubulopapillary neoplasm of the pancreas with clear cell morphology . Immunohistochemistry revealed positivity for Pan-CK , CK7 , CK8/18 , MUC1 , MUC6 , carbonic anhydrase IX , CD10 , EMA , β-catenin and e-cadherin "

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks               |   begin |   end | ner_label             |   confidence |
|---:|:-------------------------|--------:|------:|:----------------------|-------------:|
|  0 | intraductal              |      38 |    48 | CancerModifier        |     0.9998   |
|  1 | tubulopapillary          |      50 |    64 | CancerModifier        |     0.9995   |
|  2 | neoplasm of the pancreas |      66 |    89 | CancerDx              |     0.7239   |
|  3 | clear cell               |      96 |   105 | CancerModifier        |     0.96745  |
|  4 | Immunohistochemistry     |     120 |   139 | Test                  |     0.9768   |
|  5 | positivity               |     150 |   159 | Biomarker_Measurement |     0.8704   |
|  6 | Pan-CK                   |     165 |   170 | Biomarker             |     0.998    |
|  7 | CK7                      |     174 |   176 | Biomarker             |     0.9977   |
|  8 | CK8/18                   |     180 |   185 | Biomarker             |     0.9988   |
|  9 | MUC1                     |     189 |   192 | Biomarker             |     0.9965   |
| 10 | MUC6                     |     196 |   199 | Biomarker             |     0.9974   |
| 11 | carbonic anhydrase IX    |     203 |   223 | Biomarker             |     0.814033 |
| 12 | CD10                     |     227 |   230 | Biomarker             |     0.9975   |
| 13 | EMA                      |     234 |   236 | Biomarker             |     0.9985   |
| 14 | β-catenin                |     240 |   248 | Biomarker             |     0.9948   |
| 15 | e-cadherin               |     254 |   263 | Biomarker             |     0.9952   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_biomarker_pipeline|
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