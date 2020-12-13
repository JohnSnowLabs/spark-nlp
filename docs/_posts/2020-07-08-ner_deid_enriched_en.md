---
layout: model
title: Deidentification NER (Enriched)
author: John Snow Labs
name: ner_deid_enriched
class: NerDLModel
language: en
repository: clinical/models
date: 2020-07-08
tags: [clinical,ner,deidentify,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.
Deidentification NER (Enriched) is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified. The entities it annotates are Age, City, Country, Date, Doctor, Hospital, Idnum, Medicalrecord, Organization, Patient, Phone, Profession, State, Street, Username, and Zip. Clinical NER is trained with the 'embeddings_clinical' word embeddings model, so be sure to use the same embeddings in the pipeline.

## Predicted Entities 
Age, City, Country, Date, Doctor, Hospital, Idnum, Medicalrecord, Organization, Patient, Phone, Profession, State, Street, Username, Zip

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS){:.button.button-orange}[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DEMOGRAPHICS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_enriched_en_2.5.3_2.4_1594170530497.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
model = NerDLModel.pretrained("ner_deid_enriched","en","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, model, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same.""""""]})))
```

```scala
val model = NerDLModel.pretrained("ner_deid_enriched","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>

## Results

```bash
+---------------+---------+
|chunk          |ner_label|
+---------------+---------+
|Smith          |PATIENT  |
|VA Hospital    |HOSPITAL |
|Day Hospital   |HOSPITAL |
|02/04/2003     |DATE     |
|Smith          |PATIENT  |
|Day Hospital   |HOSPITAL |
|Smith          |PATIENT  |
|Smith          |PATIENT  |
|7 Ardmore Tower|HOSPITAL |
|Hart           |DOCTOR   |
|Smith          |PATIENT  |
|02/07/2003     |DATE     |
+---------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Name:           | ner_deid_enriched                |
| Type:    | NerDLModel                       |
| Compatibility:  | Spark NLP 2.4.2+                           |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | [sentence, token, word_embeddings] |
|Output labels:        | [ner]                              |
| Language:       | en                               |
| Case sensitive: | False                            |
| Dependencies:  | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on JSL enriched n2c2 2014: De-identification and Heart Disease Risk Factors Challenge datasets with `embeddings_clinical`
https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/


{:.h2_title}
## Banchmarking
```bash
|    | label            |    tp |   fp |   fn |     prec |      rec |       f1 |
|---:|:-----------------|------:|-----:|-----:|---------:|---------:|---------:|
|  0 | B-DEVICE         |     0 |    0 |    2 | 0        | 0        | 0        |
|  1 | I-AGE            |     7 |    3 |    6 | 0.7      | 0.538462 | 0.608696 |
|  2 | I-DOCTOR         |   800 |   27 |   94 | 0.967352 | 0.894855 | 0.929692 |
|  3 | I-IDNUM          |     6 |    0 |    2 | 1        | 0.75     | 0.857143 |
|  4 | B-DATE           |  1883 |   34 |   56 | 0.982264 | 0.971119 | 0.97666  |
|  5 | I-DATE           |   425 |   28 |   25 | 0.93819  | 0.944444 | 0.941307 |
|  6 | B-PHONE          |    29 |    7 |    9 | 0.805556 | 0.763158 | 0.783784 |
|  7 | B-STATE          |    87 |    4 |   11 | 0.956044 | 0.887755 | 0.920635 |
|  8 | B-CITY           |    35 |   11 |   26 | 0.76087  | 0.57377  | 0.654206 |
|  9 | I-FAX            |     0 |    0 |    4 | 0        | 0        | 0        |
| 10 | I-ORGANIZATION   |    12 |    4 |   15 | 0.75     | 0.444444 | 0.55814  |
| 11 | B-DOCTOR         |   728 |   75 |   53 | 0.9066   | 0.932138 | 0.919192 |
| 12 | I-PROFESSION     |    43 |   11 |   13 | 0.796296 | 0.767857 | 0.781818 |
| 13 | I-PHONE          |    62 |    4 |    4 | 0.939394 | 0.939394 | 0.939394 |
| 14 | I-EMAIL          |     0 |    0 |    1 | 0        | 0        | 0        |
| 15 | B-AGE            |   234 |   13 |   16 | 0.947368 | 0.936    | 0.94165  |
| 16 | B-STREET         |    20 |    7 |   16 | 0.740741 | 0.555556 | 0.634921 |
| 17 | I-ZIP            |    60 |    3 |    2 | 0.952381 | 0.967742 | 0.96     |
| 18 | I-MEDICALRECORD  |    54 |    5 |    2 | 0.915254 | 0.964286 | 0.93913  |
| 19 | B-LOCATION-OTHER |     1 |    0 |    5 | 1        | 0.166667 | 0.285714 |
| 20 | B-ZIP            |     2 |    1 |    0 | 0.666667 | 1        | 0.8      |
| 21 | B-HOSPITAL       |   256 |   23 |   66 | 0.917563 | 0.795031 | 0.851913 |
| 22 | I-STREET         |   150 |   17 |   20 | 0.898204 | 0.882353 | 0.890208 |
| 23 | B-COUNTRY        |    22 |    2 |    8 | 0.916667 | 0.733333 | 0.814815 |
| 24 | I-COUNTRY        |     1 |    0 |    0 | 1        | 1        | 1        |
| 25 | I-STATE          |     6 |    0 |    1 | 1        | 0.857143 | 0.923077 |
| 26 | B-USERNAME       |    30 |    0 |    4 | 1        | 0.882353 | 0.9375   |
| 27 | B-FAX            |     0 |    0 |    4 | 0        | 0        | 0        |
| 28 | I-HOSPITAL       |   295 |   37 |   64 | 0.888554 | 0.821727 | 0.853835 |
| 29 | I-PATIENT        |   243 |   26 |   41 | 0.903346 | 0.855634 | 0.878843 |
| 30 | B-PROFESSION     |    52 |    8 |   17 | 0.866667 | 0.753623 | 0.806202 |
| 31 | I-LOCATION-OTHER |     1 |    0 |    4 | 1        | 0.2      | 0.333333 |
| 32 | B-IDNUM          |    32 |    3 |   12 | 0.914286 | 0.727273 | 0.810127 |
| 33 | I-CITY           |    76 |   15 |   13 | 0.835165 | 0.853933 | 0.844444 |
| 34 | B-PATIENT        |   337 |   29 |   40 | 0.920765 | 0.893899 | 0.907133 |
| 35 | B-MEDICALRECORD  |    74 |    6 |    4 | 0.925    | 0.948718 | 0.936709 |
| 36 | B-ORGANIZATION   |    20 |    5 |   13 | 0.8      | 0.606061 | 0.689655 |
| 37 | Macro-average    | 6083  | 408  |  673 | 0.7976   | 0.697533 | 0.744218 |
| 38 | Micro-average    | 6083  | 408  |  673 | 0.937144 | 0.900385 | 0.918397 |
```