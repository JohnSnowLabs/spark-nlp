---
layout: model
title: Pipeline to Extract Biomarkers and their Results
author: John Snow Labs
name: ner_oncology_biomarker_pipeline
date: 2023-03-09
tags: [licensed, clinical, en, ner, oncology, biomarker]
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

This pretrained pipeline is built on the top of [ner_oncology_biomarker](https://nlp.johnsnowlabs.com/2022/11/24/ner_oncology_biomarker_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_biomarker_pipeline_en_4.3.0_3.2_1678345026649.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_biomarker_pipeline_en_4.3.0_3.2_1678345026649.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_oncology_biomarker_pipeline", "en", "clinical/models")

text = '''The results of immunohistochemical examination showed that she tested negative for CK7, synaptophysin (Syn), chromogranin A (CgA), Muc5AC, human epidermal growth factor receptor-2 (HER2), and Muc6; positive for CK20, Muc1, Muc2, E-cadherin, and p53; the Ki-67 index was about 87%.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_oncology_biomarker_pipeline", "en", "clinical/models")

val text = "The results of immunohistochemical examination showed that she tested negative for CK7, synaptophysin (Syn), chromogranin A (CgA), Muc5AC, human epidermal growth factor receptor-2 (HER2), and Muc6; positive for CK20, Muc1, Muc2, E-cadherin, and p53; the Ki-67 index was about 87%."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                               |   begin |   end | ner_label        |   confidence |
|---:|:-----------------------------------------|--------:|------:|:-----------------|-------------:|
|  0 | negative                                 |      70 |    77 | Biomarker_Result |      0.9984  |
|  1 | CK7                                      |      83 |    85 | Biomarker        |      1       |
|  2 | synaptophysin                            |      88 |   100 | Biomarker        |      0.9995  |
|  3 | Syn                                      |     103 |   105 | Biomarker        |      0.9979  |
|  4 | chromogranin A                           |     109 |   122 | Biomarker        |      0.9692  |
|  5 | CgA                                      |     125 |   127 | Biomarker        |      0.9994  |
|  6 | Muc5AC                                   |     131 |   136 | Biomarker        |      0.9987  |
|  7 | human epidermal growth factor receptor-2 |     139 |   178 | Biomarker        |      0.99876 |
|  8 | HER2                                     |     181 |   184 | Biomarker        |      1       |
|  9 | Muc6                                     |     192 |   195 | Biomarker        |      0.9999  |
| 10 | positive                                 |     198 |   205 | Biomarker_Result |      0.9987  |
| 11 | CK20                                     |     211 |   214 | Biomarker        |      1       |
| 12 | Muc1                                     |     217 |   220 | Biomarker        |      0.9999  |
| 13 | Muc2                                     |     223 |   226 | Biomarker        |      0.9999  |
| 14 | E-cadherin                               |     229 |   238 | Biomarker        |      0.9999  |
| 15 | p53                                      |     245 |   247 | Biomarker        |      1       |
| 16 | Ki-67 index                              |     254 |   264 | Biomarker        |      0.99465 |
| 17 | 87%                                      |     276 |   278 | Biomarker_Result |      0.9814  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_biomarker_pipeline|
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