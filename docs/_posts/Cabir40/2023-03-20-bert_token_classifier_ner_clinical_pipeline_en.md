---
layout: model
title: Pipeline to Detect Clinical Entities (bert_token_classifier_ner_clinical)
author: John Snow Labs
name: bert_token_classifier_ner_clinical_pipeline
date: 2023-03-20
tags: [berfortokenclassification, ner, en, licensed]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_clinical](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_clinical_pipeline_en_4.3.0_3.2_1679308200770.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_clinical_pipeline_en_4.3.0_3.2_1679308200770.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_clinical_pipeline", "en", "clinical/models")

text = '''A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_clinical_pipeline", "en", "clinical/models")

val text = "A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge ."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                     |   begin |   end | ner_label   |   confidence |
|---:|:------------------------------|--------:|------:|:------------|-------------:|
|  0 | gestational diabetes mellitus |      39 |    67 | PROBLEM     |     0.999895 |
|  1 | type two diabetes mellitus    |     128 |   153 | PROBLEM     |     0.999649 |
|  2 | T2DM                          |     157 |   160 | PROBLEM     |     0.991057 |
|  3 | HTG-induced pancreatitis      |     186 |   209 | PROBLEM     |     0.999874 |
|  4 | an acute hepatitis            |     263 |   280 | PROBLEM     |     0.999839 |
|  5 | obesity                       |     288 |   294 | PROBLEM     |     0.999873 |
|  6 | a body mass index             |     301 |   317 | TEST        |     0.974921 |
|  7 | BMI                           |     321 |   323 | TEST        |     0.972609 |
|  8 | polyuria                      |     380 |   387 | PROBLEM     |     0.999895 |
|  9 | polydipsia                    |     391 |   400 | PROBLEM     |     0.999886 |
| 10 | poor appetite                 |     404 |   416 | PROBLEM     |     0.969424 |
| 11 | vomiting                      |     424 |   431 | PROBLEM     |     0.999771 |
| 12 | amoxicillin                   |     511 |   521 | TREATMENT   |     0.995783 |
| 13 | a respiratory tract infection |     527 |   555 | PROBLEM     |     0.999406 |
| 14 | metformin                     |     570 |   578 | TREATMENT   |     0.999728 |
| 15 | glipizide                     |     582 |   590 | TREATMENT   |     0.999702 |
| 16 | dapagliflozin                 |     598 |   610 | TREATMENT   |     0.999726 |
| 17 | T2DM                          |     616 |   619 | PROBLEM     |     0.999663 |
| 18 | atorvastatin                  |     625 |   636 | TREATMENT   |     0.999727 |
| 19 | gemfibrozil                   |     642 |   652 | TREATMENT   |     0.999675 |
| 20 | HTG                           |     658 |   660 | PROBLEM     |     0.999122 |
| 21 | dapagliflozin                 |     680 |   692 | TREATMENT   |     0.999708 |
| 22 | Physical examination          |     739 |   758 | TEST        |     0.985332 |
| 23 | dry oral mucosa               |     796 |   810 | PROBLEM     |     0.991374 |
| 24 | her abdominal examination     |     830 |   854 | TEST        |     0.999292 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_clinical_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel