---
layout: model
title: Pipeline to Detect chemicals in text
author: John Snow Labs
name: ner_chemicals_pipeline
date: 2023-03-14
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

This pretrained pipeline is built on the top of [ner_chemicals](https://nlp.johnsnowlabs.com/2021/04/01/ner_chemicals_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_chemicals_pipeline_en_4.3.0_3.2_1678826470686.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_chemicals_pipeline_en_4.3.0_3.2_1678826470686.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_chemicals_pipeline", "en", "clinical/models")

text = '''The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_chemicals_pipeline", "en", "clinical/models")

val text = "The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                  |   begin |   end | ner_label   |   confidence |
|---:|:----------------------------|--------:|------:|:------------|-------------:|
|  0 | p - choloroaniline          |      40 |    57 | CHEM        |     0.935767 |
|  1 | chlorhexidine - digluconate |      90 |   116 | CHEM        |     0.855367 |
|  2 | kanamycin                   |     168 |   176 | CHEM        |     0.9824   |
|  3 | colistin                    |     180 |   187 | CHEM        |     0.9911   |
|  4 | povidone - iodine           |     193 |   209 | CHEM        |     0.8111   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_chemicals_pipeline|
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