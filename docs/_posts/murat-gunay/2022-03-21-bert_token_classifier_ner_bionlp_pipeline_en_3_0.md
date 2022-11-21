---
layout: model
title: Pipeline to Detect Cancer Genetics
author: John Snow Labs
name: bert_token_classifier_ner_bionlp_pipeline
date: 2022-03-21
tags: [licensed, ner, bionlp, en]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_bionlp](https://nlp.johnsnowlabs.com/2022/01/03/bert_token_classifier_ner_bionlp_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bionlp_pipeline_en_3.4.1_3.0_1647863601515.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("bert_token_classifier_ner_bionlp_pipeline", "en", "clinical/models")

pipeline.annotate("Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay.")
```
```scala
val pipeline = new PretrainedPipeline("bert_token_classifier_ner_bionlp_pipeline", "en", "clinical/models")

pipeline.annotate("Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay.")
```
</div>

## Results

```bash
+-------------------+----------------------+
|chunk              |ner_label             |
+-------------------+----------------------+
|erbA IRES          |Organism              |
|erbA/myb virus     |Organism              |
|erythroid cells    |Cell                  |
|bone marrow        |Multi-tissue_structure|
|blastoderm cultures|Cell                  |
|erbA/myb IRES virus|Organism              |
|erbA IRES virus    |Organism              |
|blastoderm         |Cell                  |
+-------------------+----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bionlp_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverter