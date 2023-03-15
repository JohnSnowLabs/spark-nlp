---
layout: model
title: Pipeline to Detect Chemical Compounds and Genes
author: John Snow Labs
name: ner_chemprot_clinical_pipeline
date: 2023-03-15
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

This pretrained pipeline is built on the top of [ner_chemprot_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_chemprot_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_chemprot_clinical_pipeline_en_4.3.0_3.2_1678865440862.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_chemprot_clinical_pipeline_en_4.3.0_3.2_1678865440862.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_chemprot_clinical_pipeline", "en", "clinical/models")

text = '''Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_chemprot_clinical_pipeline", "en", "clinical/models")

val text = "Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk    |   begin |   end | ner_label   |   confidence |
|---:|:-------------|--------:|------:|:------------|-------------:|
|  0 | Keratinocyte |       0 |    11 | GENE-Y      |       0.7433 |
|  1 | growth       |      13 |    18 | GENE-Y      |       0.6481 |
|  2 | factor       |      20 |    25 | GENE-Y      |       0.5693 |
|  3 | acidic       |      31 |    36 | GENE-Y      |       0.5518 |
|  4 | fibroblast   |      38 |    47 | GENE-Y      |       0.5111 |
|  5 | growth       |      49 |    54 | GENE-Y      |       0.4559 |
|  6 | factor       |      56 |    61 | GENE-Y      |       0.5213 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_chemprot_clinical_pipeline|
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