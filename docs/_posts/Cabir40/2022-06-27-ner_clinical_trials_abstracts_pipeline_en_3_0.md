---
layout: model
title: Pipeline to Extract Entities in Clinical Trial Abstracts
author: John Snow Labs
name: ner_clinical_trials_abstracts_pipeline
date: 2022-06-27
tags: [licensed, clinical, en, ner]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_clinical_trials_abstracts](https://nlp.johnsnowlabs.com/2022/06/22/ner_clinical_trials_abstracts_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_trials_abstracts_pipeline_en_3.5.3_3.0_1656313637828.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_clinical_trials_abstracts_pipeline_en_3.5.3_3.0_1656313637828.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_clinical_trials_abstracts_pipeline", "en", "clinical/models")

result = pipeline.fullAnnotate("""A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes. In a multicentre, open, randomised study, 570 patients with Type 2 diabetes, aged 34 - 80 years, were treated for 52 weeks with insulin glargine or NPH insulin given once daily at bedtime.""")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_clinical_trials_abstracts_pipeline", "en", "clinical/models")

val result = pipeline.fullAnnotate("""A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes. In a multicentre, open, randomised study, 570 patients with Type 2 diabetes, aged 34 - 80 years, were treated for 52 weeks with insulin glargine or NPH insulin given once daily at bedtime.""")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.clinical_trials_abstracts.pipe").predict("""A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes. In a multicentre, open, randomised study, 570 patients with Type 2 diabetes, aged 34 - 80 years, were treated for 52 weeks with insulin glargine or NPH insulin given once daily at bedtime.""")
```

</div>

## Results

```bash
+----------------+------------------+
|           chunk|             label|
+----------------+------------------+
|      randomised|          CTDesign|
|     multicentre|          CTDesign|
|insulin glargine|              Drug|
|     NPH insulin|              Drug|
| type 2 diabetes|DisorderOrSyndrome|
|     multicentre|          CTDesign|
|            open|          CTDesign|
|      randomised|          CTDesign|
|             570|    NumberPatients|
| Type 2 diabetes|DisorderOrSyndrome|
|              34|               Age|
|              80|               Age|
|        52 weeks|          Duration|
|insulin glargine|              Drug|
|     NPH insulin|              Drug|
|      once daily|          DrugTime|
|         bedtime|          DrugTime|
+----------------+------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_trials_abstracts_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.5.3+|
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
- NerConverter