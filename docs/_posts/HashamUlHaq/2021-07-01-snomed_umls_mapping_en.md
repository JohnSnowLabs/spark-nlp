---
layout: model
title: Snomed to UMLS Code Mapping
author: John Snow Labs
name: snomed_umls_mapping
date: 2021-07-01
tags: [snomed, umls, en, licensed, pipeline]
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

This pretrained pipeline maps SNOMED codes to UMLS codes without using any text data. You’ll just feed white space-delimited SNOMED codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_CODE_MAPPING/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/snomed_umls_mapping_en_3.1.0_2.4_1625126296775.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/snomed_umls_mapping_en_3.1.0_2.4_1625126296775.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline( 'snomed_umls_mapping','en','clinical/models')
pipeline.annotate('733187009 449433008 51264003')
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new  PretrainedPipeline( 'snomed_umls_mapping','en','clinical/models')
val result = pipeline.annotate('733187009 449433008 51264003')
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.snomed.umls").predict("""733187009 449433008 51264003""")
```

</div>

## Results

```bash
{'snomed': ['733187009', '449433008', '51264003'],
'umls': ['C4546029', 'C3164619', 'C0271267']}


Note:

|SNOMED      | Details                                                    | 
| ---------- | ----------------------------------------------------------:|
| 733187009  | osteolysis following surgical procedure on skeletal system |
| 449433008  | Diffuse stenosis of left pulmonary artery                  |
| 51264003   | Limbal AND/OR corneal involvement in vernal conjunctivitis |

| UMLS       | Details                                                    |
| ---------- | ----------------------------------------------------------:|
| C4546029   | osteolysis following surgical procedure on skeletal system |
| C3164619   | diffuse stenosis of left pulmonary artery                  |
| C0271267   | limbal and/or corneal involvement in vernal conjunctivitis |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|snomed_umls_mapping|
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