---
layout: model
title: Detect Radiology Concepts (WIP)
author: John Snow Labs
name: jsl_rd_ner_wip_greedy_clinical
date: 2021-04-01
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


Extract clinical entities from Radiology reports using pretrained NER model.


## Predicted Entities


`Kidney_Disease`, `HDL`, `Diet`, `Test`, `Imaging_Technique`, `Triglycerides`, `Obesity`, `Duration`, `Weight`, `Social_History_Header`, `ImagingTest`, `Labour_Delivery`, `Disease_Syndrome_Disorder`, `Communicable_Disease`, `Overweight`, `Units`, `Smoking`, `Score`, `Substance_Quantity`, `Form`, `Race_Ethnicity`, `Modifier`, `Hyperlipidemia`, `ImagingFindings`, `Psychological_Condition`, `OtherFindings`, `Cerebrovascular_Disease`, `Date`, `Test_Result`, `VS_Finding`, `Employment`, `Death_Entity`, `Gender`, `Oncological`, `Heart_Disease`, `Medical_Device`, `Total_Cholesterol`, `ManualFix`, `Time`, `Route`, `Pulse`, `Admission_Discharge`, `RelativeDate`, `O2_Saturation`, `Frequency`, `RelativeTime`, `Hypertension`, `Alcohol`, `Allergen`, `Fetus_NewBorn`, `Birth_Entity`, `Age`, `Respiration`, `Medical_History_Header`, `Oxygen_Therapy`, `Section_Header`, `LDL`, `Treatment`, `Vital_Signs_Header`, `Direction`, `BMI`, `Pregnancy`, `Sexually_Active_or_Sexual_Orientation`, `Symptom`, `Clinical_Dept`, `Measurements`, `Height`, `Family_History_Header`, `Substance`, `Strength`, `Injury_or_Poisoning`, `Relationship_Status`, `Blood_Pressure`, `Drug`, `Temperature`, `EKG_Findings`, `Diabetes`, `BodyPart`, `Vaccine`, `Procedure`, `Dosage`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_rd_ner_wip_greedy_clinical_en_3.0.0_3.0_1617260438155.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/jsl_rd_ner_wip_greedy_clinical_en_3.0.0_3.0_1617260438155.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("jsl_rd_ner_wip_greedy_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
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

val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("jsl_rd_ner_wip_greedy_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.jsl.wip.clinical.rd").predict("""Put your text here.""")
```

</div>


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|jsl_rd_ner_wip_greedy_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|




## Benchmarking


```bash
entity       tp      fp      fn    total  precision  recall      f1
VS_Finding    306.0   129.0   119.0    425.0     0.7034    0.72  0.7116
Direction   8717.0   678.0   616.0   9333.0     0.9278   0.934  0.9309
Respiration    224.0    28.0    18.0    242.0     0.8889  0.9256  0.9069
Cerebrovascular_Disease    149.0    57.0    64.0    213.0     0.7233  0.6995  0.7112
Family_History_Header    315.0     1.0     3.0    318.0     0.9968  0.9906  0.9937
Heart_Disease   1087.0   198.0   141.0   1228.0     0.8459  0.8852  0.8651
ImagingFindings   5568.0  1112.0  1627.0   7195.0     0.8335  0.7739  0.8026
RelativeTime    422.0   138.0   100.0    522.0     0.7536  0.8084    0.78
Strength     96.0    51.0    54.0    150.0     0.6531    0.64  0.6465
BodyPart  20155.0  1698.0  1860.0  22015.0     0.9223  0.9155  0.9189
Smoking    151.0    16.0     5.0    156.0     0.9042  0.9679   0.935
Medical_Device   8162.0   885.0   821.0   8983.0     0.9022  0.9086  0.9054
EKG_Findings    131.0    37.0    83.0    214.0     0.7798  0.6121  0.6859
Pulse    382.0    44.0    50.0    432.0     0.8967  0.8843  0.8904
Psychological_Condition    195.0    32.0    43.0    238.0      0.859  0.8193  0.8387
Triglycerides     18.0     0.0     0.0     18.0        1.0     1.0     1.0
Overweight      6.0     2.0     1.0      7.0       0.75  0.8571     0.8
Obesity     68.0     3.0     5.0     73.0     0.9577  0.9315  0.9444
Admission_Discharge    376.0    26.0    24.0    400.0     0.9353    0.94  0.9377
HDL     11.0     0.0     5.0     16.0        1.0  0.6875  0.8148
Diabetes    227.0     9.0    12.0    239.0     0.9619  0.9498  0.9558
Section_Header  13630.0   476.0   413.0  14043.0     0.9663  0.9706  0.9684
Age   1174.0   129.0    94.0   1268.0      0.901  0.9259  0.9133
O2_Saturation    122.0    34.0    29.0    151.0     0.7821  0.8079  0.7948
Drug   9391.0  1505.0   928.0  10319.0     0.8619  0.9101  0.8853
Kidney_Disease    296.0    28.0    53.0    349.0     0.9136  0.8481  0.8796
Test   3980.0   721.0   925.0   4905.0     0.8466  0.8114  0.8286
Communicable_Disease     40.0    18.0    12.0     52.0     0.6897  0.7692  0.7273
Hypertension    163.0    16.0    10.0    173.0     0.9106  0.9422  0.9261
Oxygen_Therapy    123.0    36.0    27.0    150.0     0.7736    0.82  0.7961
Test_Result   1607.0   374.0   458.0   2065.0     0.8112  0.7782  0.7944
Modifier   1229.0   435.0   593.0   1822.0     0.7386  0.6745  0.7051
BMI     21.0     4.0     7.0     28.0       0.84    0.75  0.7925
Labour_Delivery    117.0    38.0    62.0    179.0     0.7548  0.6536  0.7006
Employment    414.0    65.0    93.0    507.0     0.8643  0.8166  0.8398
Fetus_NewBorn    118.0    68.0    87.0    205.0     0.6344  0.5756  0.6036
Clinical_Dept   1937.0   189.0   133.0   2070.0     0.9111  0.9357  0.9233
Time    637.0    43.0    27.0    664.0     0.9368  0.9593  0.9479
Procedure   7578.0   953.0  1088.0   8666.0     0.8883  0.8745  0.8813
ImagingTest   1712.0   213.0   281.0   1993.0     0.8894   0.859  0.8739
Diet     79.0    44.0    82.0    161.0     0.6423  0.4907  0.5563
Oncological   1088.0   188.0   103.0   1191.0     0.8527  0.9135   0.882
LDL     20.0     7.0     1.0     21.0     0.7407  0.9524  0.8333
Symptom  15940.0  3662.0  3035.0  18975.0     0.8132  0.8401  0.8264
Temperature    240.0    28.0    25.0    265.0     0.8955  0.9057  0.9006
Vital_Signs_Header    850.0    34.0    52.0    902.0     0.9615  0.9424  0.9518
Total_Cholesterol     43.0     6.0     7.0     50.0     0.8776    0.86  0.8687
Relationship_Status     51.0     3.0     9.0     60.0     0.9444    0.85  0.8947
Blood_Pressure    353.0    18.0   117.0    470.0     0.9515  0.7511  0.8395
Injury_or_Poisoning   1003.0   311.0   241.0   1244.0     0.7633  0.8063  0.7842
Treatment    335.0    98.0    91.0    426.0     0.7737  0.7864    0.78
Pregnancy    214.0    99.0    86.0    300.0     0.6837  0.7133  0.6982
Vaccine     29.0     3.0    10.0     39.0     0.9063  0.7436  0.8169
Height    105.0    10.0    45.0    150.0      0.913     0.7  0.7925
Disease_Syndrome_Disorder   8466.0  1568.0  1533.0   9999.0     0.8437  0.8467  0.8452
Frequency   1263.0   237.0   173.0   1436.0      0.842  0.8795  0.8604
Route    219.0    35.0   144.0    363.0     0.8622  0.6033  0.7099
Duration    978.0   199.0   338.0   1316.0     0.8309  0.7432  0.7846
Death_Entity     35.0    17.0    16.0     51.0     0.6731  0.6863  0.6796
Alcohol    102.0    24.0    21.0    123.0     0.8095  0.8293  0.8193
Date    840.0    43.0    13.0    853.0     0.9513  0.9848  0.9677
Hyperlipidemia     44.0     4.0     1.0     45.0     0.9167  0.9778  0.9462
Social_History_Header    284.0     6.0    27.0    311.0     0.9793  0.9132  0.9451
ManualFix     50.0     2.0     7.0     57.0     0.9615  0.8772  0.9174
Imaging_Technique    845.0   240.0    98.0    943.0     0.7788  0.8961  0.8333
Race_Ethnicity    141.0     0.0     5.0    146.0        1.0  0.9658  0.9826
RelativeDate   1691.0   394.0   194.0   1885.0      0.811  0.8971  0.8519
Gender   6800.0   105.0   130.0   6930.0     0.9848  0.9812   0.983
Dosage    122.0    67.0    81.0    203.0     0.6455   0.601  0.6224
Medical_History_Header    486.0    10.0    19.0    505.0     0.9798  0.9624   0.971
Sexually_Active_or_Sexual_Orientation     12.0     0.0     5.0     17.0        1.0  0.7059  0.8276
Substance    102.0    11.0    22.0    124.0     0.9027  0.8226  0.8608
Weight    346.0    26.0    65.0    411.0     0.9301  0.8418  0.8838
macro      -       -       -        -         -       -     0.8038
micro      -       -       -        -         -       -     0.8793
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA3NzMzNDgyXX0=
-->