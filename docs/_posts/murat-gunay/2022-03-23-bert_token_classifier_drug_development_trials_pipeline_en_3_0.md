---
layout: model
title: Pipeline to Detect Concepts in Drug Development Trials (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_drug_development_trials_pipeline
date: 2022-03-23
tags: [licensed, ner, clinical, bertfortokenclassification, en]
task: Named Entity Recognition
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


This pretrained pipeline is built on the top of [bert_token_classifier_drug_development_trials](https://nlp.johnsnowlabs.com/2021/12/17/bert_token_classifier_drug_development_trials_en.html) model.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DRUGS_DEVELOPMENT_TRIALS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DRUGS_DEVELOPMENT_TRIALS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_drug_development_trials_pipeline_en_3.4.1_3.0_1648044113917.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_drug_development_trials_pipeline_en_3.4.1_3.0_1648044113917.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_drug_development_trials_pipeline", "en", "clinical/models")


pipeline.fullAnnotate("In June 2003, the median overall survival  with and without topotecan were 4.0 and 3.6 months, respectively. The best complete response  ( CR ) , partial response  ( PR ) , stable disease and progressive disease were observed in 23, 63, 55 and 33 patients, respectively, with  topotecan,  and 11, 61, 66 and 32 patients, respectively, without topotecan.")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_drug_development_trials_pipeline", "en", "clinical/models")


pipeline.fullAnnotate("In June 2003, the median overall survival  with and without topotecan were 4.0 and 3.6 months, respectively. The best complete response  ( CR ) , partial response  ( PR ) , stable disease and progressive disease were observed in 23, 63, 55 and 33 patients, respectively, with  topotecan,  and 11, 61, 66 and 32 patients, respectively, without topotecan.")
```
</div>


## Results


```bash
|    | chunk             | entity        |
|---:|:------------------|:--------------|
|  0 | median            | Duration      |
|  1 | overall survival  | End_Point     |
|  2 | with              | Trial_Group   |
|  3 | without topotecan | Trial_Group   |
|  4 | 4.0               | Value         |
|  5 | 3.6 months        | Value         |
|  6 | 23                | Patient_Count |
|  7 | 63                | Patient_Count |
|  8 | 55                | Patient_Count |
|  9 | 33 patients       | Patient_Count |
| 10 | topotecan         | Trial_Group   |
| 11 | 11                | Patient_Count |
| 12 | 61                | Patient_Count |
| 13 | 66                | Patient_Count |
| 14 | 32 patients       | Patient_Count |
| 15 | without topotecan | Trial_Group   |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_drug_development_trials_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.7 MB|


## Included Models


- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverter
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjQ1Mjg1MzA2LDIwNjU4NDU2OTcsMTM3NT
YyMjE3NV19
-->