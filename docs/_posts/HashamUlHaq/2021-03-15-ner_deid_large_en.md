---
layout: model
title: Detect PHI for Deidentification (Large)
author: John Snow Labs
name: ner_deid_large
date: 2021-03-15
tags: [en, licensed, ner, clinical]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 2.7.5
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Deidentification NER (Large) is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified. The entities it annotates are Age, Contact, Date, Id, Location, Name, and Profession. This model is trained with the 'embeddings_clinical' word embeddings model, so be sure to use the same embeddings in the pipeline.

We sticked to official annotation guideline (AG) for 2014 i2b2 Deid challenge while annotating new datasets for this model. All the details regarding the nuances and explanations for AG can be found here [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/)

## Predicted Entities

`AGE`, `CONTACT`, `DATE`, `ID`, `LOCATION`, `NAME`, `PROFESSION`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.1.Pretrained_Clinical_DeIdentificiation.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_large_en_2.7.5_2.4_1615837290246.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
model = NerDLModel.pretrained("ner_deid_large","en","clinical/models") \
    .setInputCols("sentence","token","embeddings") \
    .setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, model, ner_converter])               
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
input_text = ["""HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003.	HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same."""]
result = light_pipeline.fullAnnotate(input_text)

```
```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val model = NerDLModel.pretrained("ner_deid_large","en","clinical/models")
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
nlu.load("en.med_ner.deid.large").predict("""HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003.	HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same.""")
```

</div>

## Results

```bash
|    | ner_chunk       | ner_label   |
|---:|:----------------|:------------|
|  0 | Smith           | NAME        |
|  1 | VA Hospital     | LOCATION    |
|  2 | Day Hospital    | LOCATION    |
|  3 | 02/04/2003      | DATE        |
|  4 | Smith           | NAME        |
|  5 | Day Hospital    | LOCATION    |
|  6 | Smith           | NAME        |
|  7 | Smith           | NAME        |
|  8 | 7 Ardmore Tower | LOCATION    |
|  9 | Hart            | NAME        |
| 10 | Smith           | NAME        |
| 11 | 02/07/2003      | DATE        |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_large|
|Type:|ner|
|Compatibility:|Spark NLP for Healthcare 2.7.5+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Trained on augmented n2c2 2014: De-identification and Heart Disease Risk Factors Challenge.

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-NAME	 2061	 30	 29	 0.9856528	 0.9861244	 0.98588854
I-CONTACT	 146	 0	 4	 1.0	 0.97333336	 0.9864865
I-AGE	 4	 0	 1	 1.0	 0.8	 0.88888896
B-DATE	 4402	 34	 40	 0.99233544	 0.99099505	 0.99166477
I-DATE	 800	 5	 7	 0.99378884	 0.9913259	 0.9925558
I-LOCATION	 1150	 39	 53	 0.9671993	 0.95594347	 0.96153843
I-PROFESSION	 113	 10	 19	 0.9186992	 0.8560606	 0.8862745
B-NAME	 2553	 62	 58	 0.97629064	 0.9777863	 0.97703785
B-AGE	 498	 9	 12	 0.98224854	 0.9764706	 0.9793511
B-ID	 409	 11	 18	 0.97380954	 0.95784545	 0.96576154
B-PROFESSION	 132	 18	 18	 0.88	 0.88	 0.88
B-LOCATION	 1411	 75	 65	 0.94952893	 0.95596206	 0.95273465
I-ID	 33	 2	 0	 0.94285715	 1.0	 0.97058827
B-CONTACT	 158	 10	 10	 0.9404762	 0.9404762	 0.9404762
tp: 13870 fp: 305 fn: 336 labels: 14
Macro-average	 prec: 0.84393036, rec: 0.8276453, f1: 0.9357085
Micro-average	 prec: 0.97848326, rec: 0.97634804, f1: 0.9774144
```