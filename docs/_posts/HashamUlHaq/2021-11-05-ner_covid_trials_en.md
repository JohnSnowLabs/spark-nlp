---
layout: model
title: Extract entities in covid trials
author: John Snow Labs
name: ner_covid_trials
date: 2021-11-05
tags: [ner, en, clinical, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.2
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Pretrained named entity recognition deep learning model for extracting covid-related clinical terminology from covid trials.


## Predicted Entities


`Stage`, `Severity`, `Virus`, `Trial_Design`, `Trial_Phase`, `N_Patients`, `Institution`, `Statistical_Indicator`, `Section_Header`, `Cell_Type`, `Cellular_component`, `Viral_components`, `Physiological_reaction`, `Biological_molecules`, `Admission_Discharge`, `Age`, `BMI`, `Cerebrovascular_Disease`, `Date`, `Death_Entity`, `Diabetes`, `Disease_Syndrome_Disorder`, `Dosage`, `Drug_Ingredient`, `Employment`, `Frequency`, `Gender`, `Heart_Disease`, `Hypertension`, `Obesity`, `Pulse`, `Race_Ethnicity`, `Respiration`, `Route`, `Smoking`, `Time`, `Total_Cholesterol`, `Treatment`, `VS_Finding`, `Vaccine`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_COVID/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_covid_trials_en_3.2.3_3.0_1636083991325.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_covid_trials_en_3.2.3_3.0_1636083991325.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

embeddings_clinical = WordEmbeddingsModel.pretrained('embeddings_clinical', 'en', 'clinical/models') \
    .setInputCols(['sentence', 'token']) \
    .setOutputCol('embeddings')

clinical_ner = MedicalNerModel.pretrained("ner_covid_trials", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical,  clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["In December 2019 , a group of patients with the acute respiratory disease was detected in Wuhan , Hubei Province of China . A month later , a new beta-coronavirus was identified as the cause of the 2019 coronavirus infection . SARS-CoV-2 is a coronavirus that belongs to the group of β-coronaviruses of the subgenus Coronaviridae . The SARS-CoV-2 is the third known zoonotic coronavirus disease after severe acute respiratory syndrome ( SARS ) and Middle Eastern respiratory syndrome ( MERS ). The diagnosis of SARS-CoV-2 recommended by the WHO , CDC is the collection of a sample from the upper respiratory tract ( nasal and oropharyngeal exudate ) or from the lower respiratory tract such as expectoration of endotracheal aspirate and bronchioloalveolar lavage and its analysis using the test of real-time polymerase chain reaction ( qRT-PCR )."]], ["text"]))
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

val ner = MedicalNerModel.pretrained("ner_covid_trials", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val data = Seq("""In December 2019 , a group of patients with the acute respiratory disease was detected in Wuhan , Hubei Province of China . A month later , a new beta-coronavirus was identified as the cause of the 2019 coronavirus infection . SARS-CoV-2 is a coronavirus that belongs to the group of β-coronaviruses of the subgenus Coronaviridae . The SARS-CoV-2 is the third known zoonotic coronavirus disease after severe acute respiratory syndrome ( SARS ) and Middle Eastern respiratory syndrome ( MERS ). The diagnosis of SARS-CoV-2 recommended by the WHO , CDC is the collection of a sample from the upper respiratory tract ( nasal and oropharyngeal exudate ) or from the lower respiratory tract such as expectoration of endotracheal aspirate and bronchioloalveolar lavage and its analysis using the test of real-time polymerase chain reaction ( qRT-PCR ).""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.covid_trials").predict("""In December 2019 , a group of patients with the acute respiratory disease was detected in Wuhan , Hubei Province of China . A month later , a new beta-coronavirus was identified as the cause of the 2019 coronavirus infection . SARS-CoV-2 is a coronavirus that belongs to the group of β-coronaviruses of the subgenus Coronaviridae . The SARS-CoV-2 is the third known zoonotic coronavirus disease after severe acute respiratory syndrome ( SARS ) and Middle Eastern respiratory syndrome ( MERS ). The diagnosis of SARS-CoV-2 recommended by the WHO , CDC is the collection of a sample from the upper respiratory tract ( nasal and oropharyngeal exudate ) or from the lower respiratory tract such as expectoration of endotracheal aspirate and bronchioloalveolar lavage and its analysis using the test of real-time polymerase chain reaction ( qRT-PCR ).""")
```

</div>


## Results


```bash
|    | chunk                               |   begin |   end | entity                    |
|---:|:------------------------------------|--------:|------:|:--------------------------|
|  0 | December 2019                       |       3 |    15 | Date                      |
|  1 | acute respiratory disease           |      48 |    72 | Disease_Syndrome_Disorder |
|  2 | beta-coronavirus                    |     146 |   161 | Virus                     |
|  3 | 2019 coronavirus infection          |     198 |   223 | Disease_Syndrome_Disorder |
|  4 | SARS-CoV-2                          |     227 |   236 | Virus                     |
|  5 | coronavirus                         |     243 |   253 | Virus                     |
|  6 | β-coronaviruses                     |     284 |   298 | Virus                     |
|  7 | subgenus Coronaviridae              |     307 |   328 | Virus                     |
|  8 | SARS-CoV-2                          |     336 |   345 | Virus                     |
|  9 | zoonotic coronavirus disease        |     366 |   393 | Disease_Syndrome_Disorder |
| 10 | severe acute respiratory syndrome   |     401 |   433 | Disease_Syndrome_Disorder |
| 11 | SARS                                |     437 |   440 | Disease_Syndrome_Disorder |
| 12 | Middle Eastern respiratory syndrome |     448 |   482 | Disease_Syndrome_Disorder |
| 13 | MERS                                |     486 |   489 | Disease_Syndrome_Disorder |
| 14 | SARS-CoV-2                          |     511 |   520 | Virus                     |
| 15 | WHO                                 |     541 |   543 | Institution               |
| 16 | CDC                                 |     547 |   549 | Institution               |


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_covid_trials|
|Compatibility:|Healthcare NLP 3.3.2|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


This model is trained on data sampled from clinicaltrials.gov - covid trials, and annotated in-house.


## Benchmarking


```bash
label                        tp     fp    fn     prec        rec         f1         
B-Cerebrovascular_Disease    11     1     1      0.9166667   0.9166667   0.9166667  
I-Cerebrovascular_Disease    2      0     0      1.0         1.0         1.0        
I-Vaccine                    20     2     7      0.90909094  0.7407407   0.81632656 
I-N_Patients                 2      2     5      0.5         0.2857143   0.36363637 
B-Heart_Disease              32     8     11     0.8         0.74418604  0.7710843  
I-Institution                35     15    56     0.7         0.3846154   0.49645394 
B-Obesity                    8      0     0      1.0         1.0         1.0        
I-Trial_Phase                16     9     4      0.64        0.8         0.7111111  
B-Dosage                     50     28    43     0.64102566  0.53763443  0.5847953  
B-Hypertension               10     0     0      1.0         1.0         1.0        
I-Stage                      0      1     2      0.0         0.0         0.0        
I-Cell_Type                  82     31    17     0.7256637   0.82828283  0.7735849  
B-Admission_Discharge        95     1     4      0.9895833   0.959596    0.974359   
B-Date                       88     10    9      0.8979592   0.9072165   0.9025641  
I-Admission_Discharge        0      0     2      0.0         0.0         0.0        
I-Drug_Ingredient            104    71    57     0.5942857   0.6459627   0.61904764 
B-Stage                      0      2     6      0.0         0.0         0.0        
B-Cellular_component         18     22    28     0.45        0.39130434  0.41860464 
B-Total_Cholesterol          5      0     0      1.0         1.0         1.0        
I-Biological_molecules       52     39    87     0.5714286   0.37410071  0.45217392 
I-Virus                      56     26    24     0.68292683  0.7         0.69135803 
B-BMI                        9      2     2      0.8181818   0.8181818   0.8181818  
B-Drug_Ingredient            330    82    84     0.80097085  0.79710144  0.7990315  
B-Severity                   43     22    16     0.6615385   0.7288136   0.69354844 
B-Section_Header             86     27    28     0.76106197  0.75438595  0.7577093  
I-Treatment                  35     7     32     0.8333333   0.52238804  0.64220184 
I-Pulse                      1      1     0      0.5         1.0         0.6666667  
I-Respiration                1      2     0      0.33333334  1.0         0.5        
I-Section_Header             95     30    67     0.76        0.58641976  0.66202086 
I-VS_Finding                 2      2     1      0.5         0.6666667   0.57142854 
B-Death_Entity               14     3     6      0.8235294   0.7         0.7567568  
B-Statistical_Indicator      79     53    52     0.5984849   0.60305345  0.60076046 
B-Frequency                  31     8     8      0.7948718   0.7948718   0.79487187 
I-Diabetes                   3      0     0      1.0         1.0         1.0        
B-Race_Ethnicity             3      0     0      1.0         1.0         1.0        
B-Cell_Type                  90     43    35     0.6766917   0.72        0.6976744  
B-N_Patients                 37     11    9      0.7708333   0.8043478   0.787234   
B-Trial_Phase                13     4     2      0.7647059   0.8666667   0.8125     
B-Biological_molecules       171    68    122    0.71548116  0.58361775  0.6428571              
I-Hypertension               1      0     0      1.0         1.0         1.0        
I-Age                        15     0     3      1.0         0.8333333   0.90909094 
B-Employment                 75     28    21     0.7281553   0.78125     0.7537688  
B-Time                       8      0     2      1.0         0.8         0.88888896 
I-Physiological_reaction     11     11    36     0.5         0.23404256  0.3188406  
I-Viral_components           22     2     21     0.9166667   0.5116279   0.6567164  
B-Treatment                  65     8     19     0.89041096  0.77380955  0.8280255  
B-Trial_Design               26     22    19     0.5416667   0.5777778   0.5591398  
I-Severity                   6      13    4      0.31578946  0.6         0.41379312 
I-Route                      1      1     1      0.5         0.5         0.5        
B-Smoking                    3      1     0      0.75        1.0         0.85714287 
B-Diabetes                   10     0     0      1.0         1.0         1.0        
B-Gender                     21     2     6      0.9130435   0.7777778   0.84       
I-Trial_Design               41     25    13     0.6212121   0.7592593   0.6833333  
B-Virus                      105    28    34     0.7894737   0.7553957   0.77205884 
B-Vaccine                    28     9     2      0.7567568   0.93333334  0.8358209  
I-Heart_Disease              32     6     15     0.84210527  0.68085104  0.75294113 
I-Dosage                     41     32    35     0.56164384  0.5394737   0.5503355  
I-Cellular_component         12     7     19     0.6315789   0.38709676  0.48       
I-Frequency                  27     9     8      0.75        0.7714286   0.7605634  
B-Age                        14     10    11     0.5833333   0.56        0.57142854 
B-Pulse                      1      1     0      0.5         1.0         0.6666667  
I-Statistical_Indicator      40     27    44     0.5970149   0.47619048  0.52980137 
I-Date                       56     0     7      1.0         0.8888889   0.94117653 
B-Route                      38     7     12     0.84444445  0.76        0.79999995 
B-Institution                24     14    44     0.6315789   0.3529412   0.4528302  
B-Viral_components           19     7     19     0.7307692   0.5         0.59375    
B-Respiration                1      2     0      0.33333334  1.0         0.5        
I-BMI                        10     1     4      0.90909094  0.71428573  0.8000001  
B-Disease_Syndrome_Disorder  656    76    85     0.89617485  0.88529015  0.8906992  
B-VS_Finding                 23     5     1      0.8214286   0.9583333   0.8846154  
B-Physiological_reaction     9      4     26     0.6923077   0.25714287  0.37500003 
I-Disease_Syndrome_Disorder  336    50    132    0.8704663   0.71794873  0.78688526 
I-Employment                 23     9     12     0.71875     0.6571429   0.6865672  
Macro-average	             3529   1050  1482   0.7160117   0.700098    0.70796543
Micro-average	             3529   1050  1482   0.7706923   0.7042506   0.73597497
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU3ODgyMTA0OV19
-->