---
layout: model
title: Pipeline to Extract Biomarkers and Their Results
author: John Snow Labs
name: ner_oncology_biomarker_healthcare_pipeline
date: 2023-03-08
tags: [licensed, clinical, oncology, en, ner, biomarker]
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

This pretrained pipeline is built on the top of [ner_oncology_biomarker_healthcare](https://nlp.johnsnowlabs.com/2023/01/11/ner_oncology_biomarker_healthcare_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_biomarker_healthcare_pipeline_en_4.3.0_3.2_1678269721297.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_biomarker_healthcare_pipeline_en_4.3.0_3.2_1678269721297.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_oncology_biomarker_healthcare_pipeline", "en", "clinical/models")

text = '''he results of immunohistochemical examination showed that she tested negative for CK7, synaptophysin (Syn), chromogranin A (CgA), Muc5AC, human epidermal growth factor receptor-2 (HER2), and Muc6; positive for CK20, Muc1, Muc2, E-cadherin, and p53; the Ki-67 index was about 87%.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_oncology_biomarker_healthcare_pipeline", "en", "clinical/models")

val text = "he results of immunohistochemical examination showed that she tested negative for CK7, synaptophysin (Syn), chromogranin A (CgA), Muc5AC, human epidermal growth factor receptor-2 (HER2), and Muc6; positive for CK20, Muc1, Muc2, E-cadherin, and p53; the Ki-67 index was about 87%."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks                                   |   begin |   end | entities         |   confidence |
|---:|:-----------------------------------------|--------:|------:|:-----------------|-------------:|
|  0 | negative                                 |      69 |    76 | Biomarker_Result |      1       |
|  1 | CK7                                      |      82 |    84 | Biomarker        |      1       |
|  2 | synaptophysin                            |      87 |    99 | Biomarker        |      1       |
|  3 | Syn                                      |     102 |   104 | Biomarker        |      0.9999  |
|  4 | chromogranin A                           |     108 |   121 | Biomarker        |      0.99855 |
|  5 | CgA                                      |     124 |   126 | Biomarker        |      1       |
|  6 | Muc5AC                                   |     130 |   135 | Biomarker        |      0.9999  |
|  7 | human epidermal growth factor receptor-2 |     138 |   177 | Biomarker        |      0.99994 |
|  8 | HER2                                     |     180 |   183 | Biomarker        |      1       |
|  9 | Muc6                                     |     191 |   194 | Biomarker        |      1       |
| 10 | positive                                 |     197 |   204 | Biomarker_Result |      0.9997  |
| 11 | CK20                                     |     210 |   213 | Biomarker        |      1       |
| 12 | Muc1                                     |     216 |   219 | Biomarker        |      1       |
| 13 | Muc2                                     |     222 |   225 | Biomarker        |      1       |
| 14 | E-cadherin                               |     228 |   237 | Biomarker        |      0.9997  |
| 15 | p53                                      |     244 |   246 | Biomarker        |      1       |
| 16 | Ki-67 index                              |     253 |   263 | Biomarker        |      0.99865 |
| 17 | 87%                                      |     275 |   277 | Biomarker_Result |      0.828   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_biomarker_healthcare_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|533.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel