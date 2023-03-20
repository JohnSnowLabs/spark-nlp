---
layout: model
title: Pipeline to Detect Disease Mentions (MedicalBertForTokenClassification) (BERT)
author: John Snow Labs
name: bert_token_classifier_disease_mentions_tweet_pipeline
date: 2023-03-20
tags: [es, clinical, licensed, public_health, ner, token_classification, disease, tweet]
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

This pretrained pipeline is built on the top of [bert_token_classifier_disease_mentions_tweet](https://nlp.johnsnowlabs.com/2022/07/28/bert_token_classifier_disease_mentions_tweet_es_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_disease_mentions_tweet_pipeline_es_4.3.0_3.2_1679299531828.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_disease_mentions_tweet_pipeline_es_4.3.0_3.2_1679299531828.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_disease_mentions_tweet_pipeline", "es", "clinical/models")

text = '''El diagnóstico fueron varios. Principal: Neumonía en el pulmón derecho. Sinusitis de caballo, Faringitis aguda e infección de orina, también elevada. Gripe No. Estuvo hablando conmigo, sin exagerar, mas de media hora, dándome ánimo y fuerza y que sabe, porque ha visto.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_disease_mentions_tweet_pipeline", "es", "clinical/models")

val text = "El diagnóstico fueron varios. Principal: Neumonía en el pulmón derecho. Sinusitis de caballo, Faringitis aguda e infección de orina, también elevada. Gripe No. Estuvo hablando conmigo, sin exagerar, mas de media hora, dándome ánimo y fuerza y que sabe, porque ha visto."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk             |   begin |   end | ner_label   |   confidence |
|---:|:----------------------|--------:|------:|:------------|-------------:|
|  0 | Neumonía en el pulmón |      41 |    61 | ENFERMEDAD  |     0.999969 |
|  1 | Sinusitis             |      72 |    80 | ENFERMEDAD  |     0.999977 |
|  2 | Faringitis aguda      |      94 |   109 | ENFERMEDAD  |     0.999969 |
|  3 | infección de orina    |     113 |   130 | ENFERMEDAD  |     0.999969 |
|  4 | Gripe                 |     150 |   154 | ENFERMEDAD  |     0.999983 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_disease_mentions_tweet_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|es|
|Size:|462.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel