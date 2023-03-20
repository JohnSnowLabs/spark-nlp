---
layout: model
title: Pipeline to Detect Anatomical Structures (Single Entity - biobert)
author: John Snow Labs
name: ner_anatomy_coarse_biobert_pipeline
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

This pretrained pipeline is built on the top of [ner_anatomy_coarse_biobert](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_coarse_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_coarse_biobert_pipeline_en_4.3.0_3.2_1679316528376.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_coarse_biobert_pipeline_en_4.3.0_3.2_1679316528376.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_anatomy_coarse_biobert_pipeline", "en", "clinical/models")

text = '''content in the lung tissue'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_anatomy_coarse_biobert_pipeline", "en", "clinical/models")

val text = "content in the lung tissue"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk   |   begin |   end | ner_label   |   confidence |
|---:|:------------|--------:|------:|:------------|-------------:|
|  0 | lung tissue |      15 |    25 | Anatomy     |      0.99155 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_anatomy_coarse_biobert_pipeline|
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