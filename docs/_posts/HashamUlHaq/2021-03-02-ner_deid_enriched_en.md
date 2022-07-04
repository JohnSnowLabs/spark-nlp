---
layout: model
title: Detect PHI for Deidentification (Enriched)
author: John Snow Labs
name: ner_deid_enriched
date: 2021-03-02
tags: [ner, en, licensed, clinical]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 2.7.4
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

De-identification NER (Enriched) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. The entities it annotates are Age, City, Country, Date, Doctor, Hospital, Idnum, Medicalrecord, Organization, Patient, Phone, Profession, State, Street, Username, and Zip. The model is trained with the `embeddings_clinical` word embeddings model, so be sure to use the same embeddings in the pipeline.

We sticked to official annotation guideline (AG) for 2014 i2b2 Deid challenge while annotating new datasets for this model. All the details regarding the nuances and explanations for AG can be found here [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/)

## Predicted Entities

`Age`, `City`, `Country`, `Date`, `Doctor`, `Hospital`, `Idnum`, `Medicalrecord`, `Organization`, `Patient`, `Phone`, `Profession`, `State`, `Street`, `Username`, `Zip`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_enriched_en_2.7.4_2.4_1614668783590.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
   .setInputCols(["sentence", "token"])\
   .setOutputCol("embeddings")

model = NerDLModel.pretrained("ner_deid_enriched","en","clinical/models")\
   .setInputCols(["sentence","token","embeddings"])\
   .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, model, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([['HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same.']], ["text"]))

```
```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")

val model = NerDLModel.pretrained("ner_deid_enriched","en","clinical/models")
	.setInputCols("sentence","token","embeddings")
	.setOutputCol("ner")
...

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, model, ner_converter))
val data = Seq("HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same.").toDF("text")
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
|Type:|ner|
|Compatibility:|Spark NLP for Healthcare 2.7.4+|
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
label	           tp	   fp	 fn	 prec	       rec	       f1
I-AGE	           3	   0	 2	 1.0	       0.6	       0.75
I-DOCTOR	       1595	 50	 59	 0.96960485	 0.9643289	 0.96695966
I-IDNUM	         10	   2	 0	 0.8333333	 1.0	       0.90909094
B-DATE	         3665	 39	 52	 0.98947084	 0.9860102	 0.9877375
I-DATE	         591	 14	 12	 0.9768595	 0.9800995	 0.9784768
B-PHONE	         96	   8	 5	 0.9230769	 0.95049506	 0.93658537
B-STATE	         90	   1	 3	 0.989011	   0.9677419	 0.9782608
I-DEVICE	       1	   0	 0	 1.0	       1.0	       1.0
B-CITY	         125	 16	 23	 0.8865248	 0.8445946	 0.86505187
I-ORGANIZATION	 27	   0	 19	 1.0	       0.5869565	 0.739726
B-DOCTOR	       1672	 68	 73	 0.96091956	 0.9581662	 0.9595409
I-PROFESSION	   80	   15	 6	 0.84210527	 0.9302326	 0.8839779
I-PHONE	         54	   5	 3	 0.91525424	 0.94736844	 0.93103445
B-AGE	           500	 9	 16	 0.9823183	 0.96899223	 0.9756097
B-STREET	       97	   1	 2	 0.9897959	 0.97979796	 0.9847716
I-MEDICALRECORD	 10	   0	 2	 1.0	       0.8333333	 0.90909094
B-ZIP	           46	   0	 1	 1.0	       0.9787234	 0.9892473
B-HOSPITAL	     562	 38	 28	 0.93666667	 0.95254236	 0.94453776
I-STREET	       199	 1	 3	 0.995	     0.9851485	 0.9900497
B-COUNTRY	       56	   7	 11	 0.8888889	 0.8358209	 0.8615385
I-COUNTRY	       3	   2	 4	 0.6	       0.42857143	 0.5
I-STATE	         2	   0	 0	 1.0	       1.0	       1.0
B-USERNAME	     80	   0	 1	 1.0	       0.9876543	 0.99378884
I-HOSPITAL	     400	 36	 14	 0.9174312	 0.9661836	 0.9411765
I-PATIENT	       572	 56	 35	 0.91082805	 0.94233936	 0.92631584
B-PROFESSION	   85	   17	 15	 0.8333333	 0.85	       0.8415841
I-LOCATION-OTHER 1	   0	 2	 1.0	       0.33333334	 0.5
B-IDNUM	         50	   6	 5	 0.89285713	 0.90909094	 0.9009009
I-CITY	         30	   7	 2	 0.8108108	 0.9375	     0.8695652
B-PATIENT	       770	 61	 63	 0.92659444	 0.92436975	 0.9254808
B-MEDICALRECORD	 177	 6	 4	 0.9672131	 0.97790056	 0.97252744
B-ORGANIZATION	 26	   0	 30	 1.0	       0.4642857	 0.63414633

tp: 11677 fp: 467 fn: 512 labels: 38
Macro-average	 prec: 0.82731307, rec: 0.75012934, f1: 0.7868329
Micro-average	 prec: 0.9615448, rec: 0.95799494, f1: 0.95976657

```