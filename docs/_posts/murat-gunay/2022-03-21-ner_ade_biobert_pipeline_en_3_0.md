---
layout: model
title: Pipeline to Detect Adverse Drug Events (biobert)
author: John Snow Labs
name: ner_ade_biobert_pipeline
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

This pretrained pipeline is built on the top of [ner_ade_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_ade_biobert_pipeline_en_3.4.1_3.0_1647874721404.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_ade_biobert_pipeline", "en", "clinical/models")


pipeline.annotate("Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay.")
```
```scala
val pipeline = new PretrainedPipeline("ner_ade_biobert_pipeline", "en", "clinical/models")


pipeline.annotate("Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay.")
```
</div>

## Results

```bash
+--------------------------------------------+--------+
|chunks                                      |entities|
+--------------------------------------------+--------+
|5-10-fold                                   |DRUG    |
|higher transformed colony forming efficiency|ADE     |
+--------------------------------------------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ade_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|421.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter
