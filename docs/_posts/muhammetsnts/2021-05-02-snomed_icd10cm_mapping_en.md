---
layout: model
title: Snomed to ICD10 Code Mapping
author: John Snow Labs
name: snomed_icd10cm_mapping
date: 2021-05-02
tags: [snomed, icd10cm, en, licensed]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.0.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps SNOMED codes to ICD10CM codes without using any text data. You’ll just feed a comma or white space-delimited SNOMED codes and it will return the corresponding candidate ICD10CM codes as a list (multiple ICD10 codes for each Snomed code). For the time being, it supports 132K Snomed codes and 30K ICD10 codes and will be augmented & enriched in the next releases.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_CODE_MAPPING/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/snomed_icd10cm_mapping_en_3.0.2_3.0_1619955719388.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/snomed_icd10cm_mapping_en_3.0.2_3.0_1619955719388.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline( "snomed_icd10cm_mapping","en","clinical/models")
pipeline.annotate('721617001 733187009 109006')
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new PretrainedPipeline("icd10cm_snomed_mapping","en","clinical/models")
val result = pipeline.annotate('721617001 733187009 109006')
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.snomed_to_icd10cm.pipe").predict("""721617001 733187009 109006""")
```

</div>

## Results

```bash
{'snomed': ['721617001', '733187009', '109006'],
'icd10cm': ['K22.70, C15.5',
'M89.59, M89.50, M96.89',
'F41.9, F40.10, F94.8, F93.0, F40.8, F93.8']}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|snomed_icd10cm_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.0.2+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- TokenizerModel
- LemmatizerModel
- Finisher