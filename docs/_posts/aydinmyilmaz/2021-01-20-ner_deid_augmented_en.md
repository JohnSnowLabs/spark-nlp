---
layout: model
title: Detect PHI for Deidentification (Augmented)
author: John Snow Labs
name: ner_deid_augmented
date: 2021-01-20
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 2.7.1
spark_version: 2.4
tags: [en, deidentify, ner, clinical, licensed]
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Deidentification NER (Augmented) is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified.

We sticked to official annotation guideline (AG) for 2014 i2b2 Deid challenge while annotating new datasets for this model. All the details regarding the nuances and explanations for AG can be found here [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/)

## Predicted Entities

`AGE`, `CONTACT`, `DATE`, `ID`, `LOCATION`, `NAME`, `PROFESSION`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/3de6f25c23cd487d829ac3ce444ef19cfbe02631/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentificiation.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_augmented_en_2.7.1_2.4_1611145829422.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_augmented_en_2.7.1_2.4_1611145829422.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

This model is trained with the ‘embeddings_clinical’ word embeddings, so be sure to use the same embeddings within the pipeline in addition to document assembler, sentence detector, tokenizer and ner converter .

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

ner_model = NerDLModel.pretrained("ner_deid_augmented","en","clinical/models")\
	.setInputCols(["sentence","token","word_embeddings"])\
	.setOutputCol("ner")

ner_converter = NerConverter() \
  	.setInputCols(['document', 'token', 'ner']) \
  	.setOutputCol('ner_chunk')

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, ner_model, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([['HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital, Dr. John Green (2347165768). He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same. ']], ["text"]))
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

val ner_model = NerDLModel.pretrained("ner_deid_augmented","en","clinical/models")
	.setInputCols(Array("sentence","token","word_embeddings"))
	.setOutputCol("ner")

val ner_converter = new NerConverter()
  	.setInputCols(Array("document", "token", "ner"))
  	.setOutputCol("ner_chunk")

val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner_model, ner_converter))

val result = pipeline.fit(Seq.empty[String]).transform(data)

val results = LightPipeline(model).fullAnnotate("""HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital, Dr. John Green (2347165768). He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same.""")
```



{:.nlu-block}

```python
import nlu
nlu.load("en.med_ner.deid.augmented").predict("""HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital, Dr. John Green (2347165768). He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same. """)
```

</div>

## Results

```bash
+---------------+---------+
|chunk          |ner_label|
+---------------+---------+
|Smith          |NAME     |
|VA Hospital    |LOCATION |
|John Green     |NAME     |
|2347165768     |ID       |
|Day Hospital   |LOCATION |
|02/04/2003     |DATE     |
|Smith          |NAME     |
|Day Hospital   |LOCATION |
|Smith          |NAME     |
|Smith          |NAME     |
|7 Ardmore Tower|LOCATION |
|Hart           |NAME     |
|Smith          |NAME     |
|02/07/2003     |DATE     |
+---------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_augmented|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Data Source

Trained on plain n2c2 2014: De-identification and Heart Disease Risk Factors Challenge datasets with embeddings_clinical https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/

## Benchmarking

```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|--------------:|------:|------:|------:|---------:|---------:|---------:|
|  0 | I-NAME        |  1096 |    47 |    80 | 0.95888  | 0.931973 | 0.945235 |
|  1 | I-CONTACT     |    93 |     0 |     4 | 1        | 0.958763 | 0.978947 |
|  2 | I-AGE         |     3 |     1 |     6 | 0.75     | 0.333333 | 0.461538 |
|  3 | B-DATE        |  2078 |    42 |    52 | 0.980189 | 0.975587 | 0.977882 |
|  4 | I-DATE        |   474 |    39 |    25 | 0.923977 | 0.9499   | 0.936759 |
|  5 | I-LOCATION    |   755 |    68 |    76 | 0.917375 | 0.908544 | 0.912938 |
|  6 | I-PROFESSION  |    78 |     8 |     9 | 0.906977 | 0.896552 | 0.901734 |
|  7 | B-NAME        |  1182 |   101 |    36 | 0.921278 | 0.970443 | 0.945222 |
|  8 | B-AGE         |   259 |    10 |    11 | 0.962825 | 0.959259 | 0.961039 |
|  9 | B-ID          |   146 |     8 |    11 | 0.948052 | 0.929936 | 0.938907 |
| 10 | B-PROFESSION  |    76 |     9 |    21 | 0.894118 | 0.783505 | 0.835165 |
| 11 | B-LOCATION    |   556 |    87 |    71 | 0.864697 | 0.886762 | 0.875591 |
| 12 | I-ID          |    64 |     8 |     3 | 0.888889 | 0.955224 | 0.920863 |
| 13 | B-CONTACT     |    40 |     7 |     5 | 0.851064 | 0.888889 | 0.869565 |
| 14 | Macro-average |  6900 |   435 |   410 | 0.912023 | 0.880619 | 0.896046 |
| 15 | Micro-average |  6900 |   435 |   410 | 0.940695 | 0.943912 | 0.942301 |
```
