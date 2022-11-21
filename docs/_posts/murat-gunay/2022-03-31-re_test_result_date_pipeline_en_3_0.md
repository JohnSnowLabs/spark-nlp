---
layout: model
title: RE Pipeline between Tests, Results, and Dates
author: John Snow Labs
name: re_test_result_date_pipeline
date: 2022-03-31
tags: [licensed, clinical, relation_extraction, tests, results, dates, en]
task: Relation Extraction
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


This pretrained pipeline is built on the top of [re_test_result_date](https://nlp.johnsnowlabs.com/2021/02/24/re_test_result_date_en.html) model.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_DATE/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_CLINICAL_DATE.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_test_result_date_pipeline_en_3.4.1_3.0_1648734076557.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("re_test_result_date_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("He was advised chest X-ray or CT scan after checking his SpO2 which was <= 93%")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("re_test_result_date_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("He was advised chest X-ray or CT scan after checking his SpO2 which was <= 93%")
```
</div>


## Results


```bash
| index | relations    | entity1      | chunk1              | entity2      |  chunk2 |
|-------|--------------|--------------|---------------------|--------------|---------|
| 0     | O            | TEST         | chest X-ray         | MEASUREMENTS |  93%    | 
| 1     | O            | TEST         | CT scan             | MEASUREMENTS |  93%    |
| 2     | is_result_of | TEST         | SpO2                | MEASUREMENTS |  93%    |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|re_test_result_date_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|


## Included Models


- DocumentAssembler
- SentenceDetector
- TokenizerModel
- PerceptronModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- DependencyParserModel
- RelationExtractionModel
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1MDk1OTUwNjYsLTEwMDg0MTMxNywtMj
g2MjE1ODE0LC0yMzIzOTUyNTcsMjEwODgzMDk1MSwxODgxMDA3
ODE3LC03OTUzOTI3NzddfQ==
-->