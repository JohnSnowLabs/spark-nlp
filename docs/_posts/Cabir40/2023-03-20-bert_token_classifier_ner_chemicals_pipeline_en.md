---
layout: model
title: Pipeline to Detect Chemicals in Medical text (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_chemicals_pipeline
date: 2023-03-20
tags: [berfortokenclassification, ner, chemicals, en, licensed]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_chemicals](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_chemicals_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemicals_pipeline_en_4.3.0_3.2_1679306458020.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemicals_pipeline_en_4.3.0_3.2_1679306458020.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_chemicals_pipeline", "en", "clinical/models")

text = '''The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. "A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_chemicals_pipeline", "en", "clinical/models")

val text = "The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. "A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                   |   begin |   end | ner_label   |   confidence |
|---:|:----------------------------|--------:|------:|:------------|-------------:|
|  0 | p - choloroaniline          |      40 |    57 | CHEM        |     0.999986 |
|  1 | chlorhexidine - digluconate |      90 |   116 | CHEM        |     0.999989 |
|  2 | kanamycin                   |     169 |   177 | CHEM        |     0.999985 |
|  3 | colistin                    |     181 |   188 | CHEM        |     0.999982 |
|  4 | povidone - iodine           |     194 |   210 | CHEM        |     0.99998  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_chemicals_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel