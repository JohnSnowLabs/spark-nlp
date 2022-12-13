---
layout: model
title: RxNorm to UMLS Code Mapping
author: John Snow Labs
name: rxnorm_umls_mapping
date: 2021-07-01
tags: [rxnorm, umls, en, licensed, pipeline]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.1.0
spark_version: 2.4
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps RxNorm codes to UMLS codes without using any text data. You’ll just feed white space-delimited RxNorm codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_CODE_MAPPING/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rxnorm_umls_mapping_en_3.1.0_2.4_1625126295049.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/rxnorm_umls_mapping_en_3.1.0_2.4_1625126295049.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline( "rxnorm_umls_mapping","en","clinical/models")
pipeline.annotate("1161611 315677 343663")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new PretrainedPipeline( "rxnorm_umls_mapping","en","clinical/models")
val result = pipeline.annotate("1161611 315677 343663")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.rxnorm.umls").predict("""1161611 315677 343663""")
```

</div>

## Results

```bash
{'rxnorm': ['1161611', '315677', '343663'],
'umls': ['C3215948', 'C0984912', 'C1146501']}


Note:

| RxNorm     | Details                  | 
| ---------- | ------------------------:|
| 1161611    |  metformin Pill          |
| 315677     | cimetidine 100 mg        |
| 343663     | insulin lispro 50 UNT/ML |

| UMLS       | Details                  |
| ---------- | ------------------------:|
| C3215948   | metformin pill           |
| C0984912   | cimetidine 100 mg        |
| C1146501   | insulin lispro 50 unt/ml |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rxnorm_umls_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.1.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- TokenizerModel
- LemmatizerModel
- Finisher