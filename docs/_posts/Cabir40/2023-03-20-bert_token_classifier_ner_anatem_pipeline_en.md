---
layout: model
title: Pipeline to Detect Anatomical Structures in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_anatem_pipeline
date: 2023-03-20
tags: [en, ner, clinical, licensed, bertfortokenclassification]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_anatem](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_anatem_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_anatem_pipeline_en_4.3.0_3.2_1679303142191.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_anatem_pipeline_en_4.3.0_3.2_1679303142191.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_anatem_pipeline", "en", "clinical/models")

text = '''Malignant cells often display defects in autophagy, an evolutionarily conserved pathway for degrading long-lived proteins and cytoplasmic organelles. However, as yet, there is no genetic evidence for a role of autophagy genes in tumor suppression. The beclin 1 autophagy gene is monoallelically deleted in 40 - 75 % of cases of human sporadic breast, ovarian, and prostate cancer.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_anatem_pipeline", "en", "clinical/models")

val text = "Malignant cells often display defects in autophagy, an evolutionarily conserved pathway for degrading long-lived proteins and cytoplasmic organelles. However, as yet, there is no genetic evidence for a role of autophagy genes in tumor suppression. The beclin 1 autophagy gene is monoallelically deleted in 40 - 75 % of cases of human sporadic breast, ovarian, and prostate cancer."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk              |   begin |   end | ner_label   |   confidence |
|---:|:-----------------------|--------:|------:|:------------|-------------:|
|  0 | Malignant cells        |       0 |    14 | Anatomy     |     0.999951 |
|  1 | cytoplasmic organelles |     126 |   147 | Anatomy     |     0.999937 |
|  2 | tumor                  |     229 |   233 | Anatomy     |     0.999871 |
|  3 | breast                 |     343 |   348 | Anatomy     |     0.999842 |
|  4 | ovarian                |     351 |   357 | Anatomy     |     0.99998  |
|  5 | prostate cancer        |     364 |   378 | Anatomy     |     0.999968 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_anatem_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel