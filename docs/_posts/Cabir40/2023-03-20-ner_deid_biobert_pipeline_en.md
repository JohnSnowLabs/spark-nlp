---
layout: model
title: Pipeline to Detect PHI in medical text (biobert)
author: John Snow Labs
name: ner_deid_biobert_pipeline
date: 2023-03-20
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

This pretrained pipeline is built on the top of [ner_deid_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_biobert_pipeline_en_4.3.0_3.2_1679310594035.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_biobert_pipeline_en_4.3.0_3.2_1679310594035.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_deid_biobert_pipeline", "en", "clinical/models")

text = '''A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_deid_biobert_pipeline", "en", "clinical/models")

val text = "A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                     |   begin |   end | ner_label   |   confidence |
|---:|:------------------------------|--------:|------:|:------------|-------------:|
|  0 | 2093-01-13                    |      17 |    26 | DATE        |      0.981   |
|  1 | David Hale                    |      29 |    38 | NAME        |      0.77585 |
|  2 | Hendrickson                   |      53 |    63 | NAME        |      0.9666  |
|  3 | Ora                           |      66 |    68 | LOCATION    |      0.8723  |
|  4 | Oliveira                      |      91 |    98 | LOCATION    |      0.7785  |
|  5 | Cocke County Baptist Hospital |     114 |   142 | LOCATION    |      0.792   |
|  6 | Keats Street                  |     150 |   161 | LOCATION    |      0.77305 |
|  7 | Phone                         |     164 |   168 | LOCATION    |      0.7083  |
|  8 | Brothers                      |     253 |   260 | LOCATION    |      0.9447  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel