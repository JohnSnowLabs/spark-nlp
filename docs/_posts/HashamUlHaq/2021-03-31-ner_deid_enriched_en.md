---
layout: model
title: Detect PHI for Deidentification (Enriched)
author: John Snow Labs
name: ner_deid_enriched
date: 2021-03-31
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.Deidentification NER (Enriched) is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified. The entities it annotates are Age, City, Country, Date, Doctor, Hospital, Idnum, Medicalrecord, Organization, Patient, Phone, Profession, State, Street, Username, and Zip. Clinical NER is trained with the 'embeddings_clinical' word embeddings model, so be sure to use the same embeddings in the pipeline.

We sticked to official annotation guideline (AG) for 2014 i2b2 Deid challenge while annotating new datasets for this model. All the details regarding the nuances and explanations for AG can be found here [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/)

## Predicted Entities

``Age``, ``City``, ``Country``, ``Date``, ``Doctor``, ``Hospital``, ``Idnum``, ``Medicalrecord``, ``Organization``, ``Patient``, ``Phone``, ``Profession``, ``State``, ``Street``, ``Username``, ``Zip``

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_enriched_en_3.0.0_3.0_1617208426129.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("ner_deid_enriched","en","clinical/models")\
  .setInputCols(["sentence","token","embeddings"])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
 	  .setInputCols(["sentence", "token", "ner"])\
 	  .setOutputCol("ner_chunk")
    
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([['HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same.']], ["text"]))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
	.setInputCols(Array("sentence", "token"))
	.setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_deid_enriched","en","clinical/models")
	.setInputCols("sentence","token","embeddings")
	.setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq("""HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same.""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.deid.enriched").predict("""HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same.""")
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
|---|---|
|Model Name:|ner_deid_enriched|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Trained on JSL enriched n2c2 2014: De-identification and Heart Disease Risk Factors Challenge datasets with `embeddings_clinical`
https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/

## Benchmarking

```bash
label                tp    fp    fn      prec       rec        f1
I-AGE                 7     3     6  0.7       0.538462  0.608696
I-DOCTOR            800    27    94  0.967352  0.894855  0.929692
I-IDNUM               6     0     2  1         0.75      0.857143
B-DATE             1883    34    56  0.982264  0.971119  0.97666 
I-DATE              425    28    25  0.93819   0.944444  0.941307
B-PHONE              29     7     9  0.805556  0.763158  0.783784
B-STATE              87     4    11  0.956044  0.887755  0.920635
B-CITY               35    11    26  0.76087   0.57377   0.654206
I-ORGANIZATION       12     4    15  0.75      0.444444  0.55814 
B-DOCTOR            728    75    53  0.9066    0.932138  0.919192
I-PROFESSION         43    11    13  0.796296  0.767857  0.781818
I-PHONE              62     4     4  0.939394  0.939394  0.939394
B-AGE               234    13    16  0.947368  0.936     0.94165 
B-STREET             20     7    16  0.740741  0.555556  0.634921
I-ZIP                60     3     2  0.952381  0.967742  0.96    
I-MEDICALRECORD      54     5     2  0.915254  0.964286  0.93913 
B-ZIP                 2     1     0  0.666667  1         0.8     
B-HOSPITAL          256    23    66  0.917563  0.795031  0.851913
I-STREET            150    17    20  0.898204  0.882353  0.890208
B-COUNTRY            22     2     8  0.916667  0.733333  0.814815
I-COUNTRY             1     0     0  1         1         1       
I-STATE               6     0     1  1         0.857143  0.923077
B-USERNAME           30     0     4  1         0.882353  0.9375  
I-HOSPITAL          295    37    64  0.888554  0.821727  0.853835
I-PATIENT           243    26    41  0.903346  0.855634  0.878843
B-PROFESSION         52     8    17  0.866667  0.753623  0.806202
B-IDNUM              32     3    12  0.914286  0.727273  0.810127
I-CITY               76    15    13  0.835165  0.853933  0.844444
B-PATIENT           337    29    40  0.920765  0.893899  0.907133
B-MEDICALRECORD      74     6     4  0.925     0.948718  0.936709
B-ORGANIZATION       20     5    13  0.8       0.606061  0.689655
Macro-average      6083   408   673  0.7976    0.697533  0.744218
Micro-average      6083   408   673  0.937144  0.900385  0.918397
```