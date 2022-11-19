---
layout: model
title: Pipeline to Detect chemicals in text
author: John Snow Labs
name: ner_chemicals_pipeline
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

This pretrained pipeline is built on the top of [ner_chemicals](https://nlp.johnsnowlabs.com/2021/04/01/ner_chemicals_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_chemicals_pipeline_en_3.4.1_3.0_1647869797628.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
pipeline = PretrainedPipeline("ner_chemicals_pipeline", "en", "clinical/models")


pipeline.annotate("The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis.")
```
```scala
val pipeline = new PretrainedPipeline("ner_chemicals_pipeline", "en", "clinical/models")


pipeline.annotate("The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis.")
```
</div>

## Results

```bash
+---------------------------+--------+
|chunks                     |entities|
+---------------------------+--------+
|p - choloroaniline         |CHEM    |
|chlorhexidine - digluconate|CHEM    |
|kanamycin                  |CHEM    |
|colistin                   |CHEM    |
|povidone - iodine          |CHEM    |
+---------------------------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_chemicals_pipeline|
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
