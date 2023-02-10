---
layout: model
title: ICD10 to UMLS Code Mapping
author: John Snow Labs
name: icd10cm_umls_mapping
date: 2021-07-01
tags: [icd10cm, umls, en, licensed, pipeline]
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

This pretrained pipeline maps ICD10CM codes to UMLS codes without using any text data. You’ll just feed white space-delimited ICD10CM codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_CODE_MAPPING/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10cm_umls_mapping_en_3.1.0_2.4_1625126281405.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/icd10cm_umls_mapping_en_3.1.0_2.4_1625126281405.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline( "icd10cm_umls_mapping","en","clinical/models")
pipeline.annotate(["M8950", "R822", "R0901"])
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new PretrainedPipeline("icd10cm_umls_mapping","en","clinical/models")
val result = pipeline.annotate(["M8950", "R822", "R0901"])
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.icd10cm.umls").predict("""M8950 R822 R0901""")
```

</div>

## Results

```bash
{'icd10cm': ['M89.50', 'R82.2', 'R09.01'],
'umls': ['C4721411', 'C0159076', 'C0004044']}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10cm_umls_mapping|
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