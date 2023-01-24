---
layout: model
title: RE Pipeline between Problem, Test, and Findings in Reports
author: John Snow Labs
name: re_test_problem_finding_pipeline
date: 2022-03-31
tags: [licensed, clinical, relation_extraction, problem, test, findings, en]
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


This pretrained pipeline is built on the top of [re_test_problem_finding](https://nlp.johnsnowlabs.com/2021/04/19/re_test_problem_finding_en.html) model.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_RADIOLOGY/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_RADIOLOGY.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_test_problem_finding_pipeline_en_3.4.1_3.0_1648733292407.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_test_problem_finding_pipeline_en_3.4.1_3.0_1648733292407.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("re_test_problem_finding_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("Targeted biopsy of this lesion for histological correlation should be considered.")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("re_test_problem_finding_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("Targeted biopsy of this lesion for histological correlation should be considered.")
```
</div>


## Results


```bash
| index | relations    | entity1      | chunk1              | entity2      |  chunk2 |
|-------|--------------|--------------|---------------------|--------------|---------|
| 0     | 1            | PROCEDURE    | biopsy              | SYMPTOM      |  lesion | 
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|re_test_problem_finding_pipeline|
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
eyJoaXN0b3J5IjpbLTY2ODIwNTI0NiwtMzk5NjI4NzU4LC0xOT
gzOTYyMDM3XX0=
-->