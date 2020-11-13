---
layout: model
title: Deidentification NER (Large)
author: John Snow Labs
name: ner_deid_large
class: NerDLModel
language: en
repository: clinical/models
date: 2020-07-22
tags: [clinical,ner,deidentify,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description

Deidentification NER (Large) is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified. The entities it annotates are Age, Contact, Date, Id, Location, Name, and Profession. This model is trained with the `'embeddings_clinical'` word embeddings model, so be sure to use the same embeddings in the pipeline.

## Predicted Entities 
Age, Contact, Date, Id, Location, Name, Profession

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS){:.button.button-orange}[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DEMOGRAPHICS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_large_en_2.5.3_2.4_1595427435246.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...

model = NerDLModel.pretrained("ner_deid_large","en","clinical/models")
    .setInputCols("sentence","token","word_embeddings")
    .setOutputCol("ner")

...

nlp_pipeline = Pipeline(stages=[document_assembler,
                                sentence_detector,
                                tokenizer,
                                word_embeddings,
                                model,
                                ner_converter])
                                
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

input_text = [
    """HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003.

HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same."""
]
result = pipeline_model.transform(spark.createDataFrame(pd.DataFrame({"text": input_text})))
```

```scala
...

val model = NerDLModel.pretrained("ner_deid_large","en","clinical/models")
    .setInputCols("sentence","token","word_embeddings")
    .setOutputCol("ner")
    
...

val pipeline = new Pipeline().setStages(Array(
                                document_assembler,
                                sentence_detector,
                                tokenizer,
                                word_embeddings,
                                model,
                                ner_converter))

val result = pipeline.fit(Seq.empty["""HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003.

HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same."""].toDS.toDF("text")).transform(data)
 
```
</div>

## Results
```bash
+---------------+---------+
|chunk          |ner_label|
+---------------+---------+
|Smith          |PATIENT  |
|Smith          |PATIENT  |
|VA Hospital    |HOSPITAL |
|Day Hospital   |HOSPITAL |
|02/04/2003     |DATE     |
|Smith          |PATIENT  |
|Day Hospital   |HOSPITAL |
|Smith          |PATIENT  |
|Smith          |PATIENT  |
|7 Ardmore Tower|STREET   |
|Hart           |DOCTOR   |
|Smith          |PATIENT  |
|02/07/2003     |DATE     |
+---------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Name:           | ner_deid_large                   |
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
Trained on plain n2c2 2014: De-identification and Heart Disease Risk Factors Challenge datasets with `embeddings_clinical`
https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/

## Benchmarking
```bash
|    | label           |     tp |    fp |   fn |     prec |      rec |        f1 |
|---:|----------------:|-------:|------:|-----:|---------:|---------:|----------:|
|  0 | I-TIME          |     82 |    12 |   45 | 0.87234  | 0.645669 | 0.742081  |
|  1 | I-TREATMENT     |   2580 |   439 |  535 | 0.854588 | 0.82825  | 0.841213  |
|  2 | B-OCCURRENCE    |   1548 |   680 |  945 | 0.694793 | 0.620939 | 0.655793  |
|  3 | I-DURATION      |    366 |   183 |   99 | 0.666667 | 0.787097 | 0.721893  |
|  4 | B-DATE          |    847 |   151 |  138 | 0.848697 | 0.859898 | 0.854261  |
|  5 | I-DATE          |    921 |   191 |  196 | 0.828237 | 0.82453  | 0.82638   |
|  6 | B-ADMISSION     |    105 |   102 |   15 | 0.507246 | 0.875    | 0.642202  |
|  7 | I-PROBLEM       |   5238 |   902 |  823 | 0.853094 | 0.864214 | 0.858618  |
|  8 | B-CLINICAL_DEPT |    613 |   130 |  119 | 0.825034 | 0.837432 | 0.831187  |
|  9 | B-TIME          |     36 |     8 |   24 | 0.818182 | 0.6      | 0.692308  |
| 10 | I-CLINICAL_DEPT |   1273 |   210 |  137 | 0.858395 | 0.902837 | 0.880055  |
| 11 | B-PROBLEM       |   3717 |   608 |  591 | 0.859422 | 0.862813 | 0.861114  |
| 12 | I-TEST          |   2304 |   384 |  361 | 0.857143 | 0.86454  | 0.860826  |
| 13 | B-TEST          |   1870 |   372 |  300 | 0.834077 | 0.861751 | 0.847688  |
| 14 | B-TREATMENT     |   2767 |   437 |  513 | 0.863608 | 0.843598 | 0.853485  |
| 15 | B-EVIDENTIAL    |    394 |   109 |  201 | 0.7833   | 0.662185 | 0.717669  |
| 16 | B-DURATION      |    236 |   119 |  105 | 0.664789 | 0.692082 | 0.678161  |
| 17 | B-FREQUENCY     |    117 |    20 |   79 | 0.854015 | 0.596939 | 0.702703  |
| 18 | Macro-average   | 25806  | 5821  | 6342 | 0.735285 | 0.677034 | 0.704959  |
| 19 | Micro-average   | 25806  | 5821  | 6342 | 0.815948 | 0.802725 | 0.809283  |
```