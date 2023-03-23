---
layout: model
title: Pipeline to Detect Diseases in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_bc5cdr_disease_pipeline
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_bc5cdr_disease](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_bc5cdr_disease_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc5cdr_disease_pipeline_en_4.3.0_3.2_1679302082722.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc5cdr_disease_pipeline_en_4.3.0_3.2_1679302082722.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_bc5cdr_disease_pipeline", "en", "clinical/models")

text = '''Indomethacin resulted in histopathologic findings typical of interstitial cystitis, such as leaky bladder epithelium and mucosal mastocytosis. The true incidence of nonsteroidal anti-inflammatory drug-induced cystitis in humans must be clarified by prospective clinical trials. An open-label phase II study of low-dose thalidomide in androgen-independent prostate cancer.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_bc5cdr_disease_pipeline", "en", "clinical/models")

val text = "Indomethacin resulted in histopathologic findings typical of interstitial cystitis, such as leaky bladder epithelium and mucosal mastocytosis. The true incidence of nonsteroidal anti-inflammatory drug-induced cystitis in humans must be clarified by prospective clinical trials. An open-label phase II study of low-dose thalidomide in androgen-independent prostate cancer."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk             |   begin |   end | ner_label   |   confidence |
|---:|:----------------------|--------:|------:|:------------|-------------:|
|  0 | interstitial cystitis |      61 |    81 | DISEASE     |     0.999746 |
|  1 | mastocytosis          |     129 |   140 | DISEASE     |     0.999132 |
|  2 | cystitis              |     209 |   216 | DISEASE     |     0.999912 |
|  3 | prostate cancer       |     355 |   369 | DISEASE     |     0.999781 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bc5cdr_disease_pipeline|
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