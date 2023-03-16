---
layout: model
title: Pipeline to Detect Drugs and Proteins
author: John Snow Labs
name: ner_drugprot_clinical_pipeline
date: 2023-03-14
tags: [ner, clinical, drugprot, en, licensed]
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

This pretrained pipeline is built on the top of [ner_drugprot_clinical](https://nlp.johnsnowlabs.com/2021/12/20/ner_drugprot_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_drugprot_clinical_pipeline_en_4.3.0_3.2_1678777770925.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_drugprot_clinical_pipeline_en_4.3.0_3.2_1678777770925.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_drugprot_clinical_pipeline", "en", "clinical/models")

text = '''Anabolic effects of clenbuterol on skeletal muscle are mediated by beta 2-adrenoceptor activation.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_drugprot_clinical_pipeline", "en", "clinical/models")

val text = "Anabolic effects of clenbuterol on skeletal muscle are mediated by beta 2-adrenoceptor activation."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks          |   begin |   end | ner_label   |   confidence |
|---:|:--------------------|--------:|------:|:------------|-------------:|
|  0 | clenbuterol         |      20 |    30 | CHEMICAL    |      0.9691  |
|  1 | beta 2-adrenoceptor |      67 |    85 | GENE        |      0.89855 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_drugprot_clinical_pipeline|
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