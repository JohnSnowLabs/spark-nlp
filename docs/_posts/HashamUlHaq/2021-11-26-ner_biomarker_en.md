---
layout: model
title: Extraction of biomarker information
author: John Snow Labs
name: ner_biomarker
date: 2021-11-26
tags: [en, ner, clinical, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.3
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is trained to extract biomarkers, therapies, oncological, and other general concepts from text.


## Predicted Entities


`Oncogenes`, `Tumor_Finding`, `UnspecificTherapy`, `Ethnicity`, `Age`, `ResponseToTreatment`, `Biomarker`, `HormonalTherapy`, `Staging`, `Drug`, `CancerDx`, `Radiotherapy`, `CancerSurgery`, `TargetedTherapy`, `PerformanceStatus`, `CancerModifier`, `Radiological_Test_Result`, `Biomarker_Measurement`, `Metastasis`, `Radiological_Test`, `Chemotherapy`, `Test`, `Dosage`, `Test_Result`, `Immunotherapy`, `Date`, `Gender`, `Prognostic_Biomarkers`, `Duration`, `Predictive_Biomarkers`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_BIOMARKER/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_biomarker_en_3.3.3_3.0_1637935088644.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_biomarker_en_3.3.3_3.0_1637935088644.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

clinical_ner = MedicalNerModel.pretrained("ner_biomarker", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	  .setInputCols(["sentence", "token", "ner"])\
 	  .setOutputCol("ner_chunk")
    
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical,  clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["Here , we report the first case of an intraductal tubulopapillary neoplasm of the pancreas with clear cell morphology . Immunohistochemistry revealed positivity for Pan-CK , CK7 , CK8/18 , MUC1 , MUC6 , carbonic anhydrase IX , CD10 , EMA , β-catenin and e-cadherin "]], ["text"]))


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

val ner = MedicalNerModel.pretrained("ner_biomarker", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")
    
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val data = Seq("""Here , we report the first case of an intraductal tubulopapillary neoplasm of the pancreas with clear cell morphology . Immunohistochemistry revealed positivity for Pan-CK , CK7 , CK8/18 , MUC1 , MUC6 , carbonic anhydrase IX , CD10 , EMA , β-catenin and e-cadherin """).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.biomarker").predict("""Here , we report the first case of an intraductal tubulopapillary neoplasm of the pancreas with clear cell morphology . Immunohistochemistry revealed positivity for Pan-CK , CK7 , CK8/18 , MUC1 , MUC6 , carbonic anhydrase IX , CD10 , EMA , β-catenin and e-cadherin """)
```

</div>


## Results


```bash
|    | ner_chunk                | entity                |   confidence |
|---:|:-------------------------|:----------------------|-------------:|
|  0 | intraductal              | CancerModifier        |     0.9934   |
|  1 | tubulopapillary          | CancerModifier        |     0.6403   |
|  2 | neoplasm of the pancreas | CancerDx              |     0.758825 |
|  3 | clear cell               | CancerModifier        |     0.9633   |
|  4 | Immunohistochemistry     | Test                  |     0.9534   |
|  5 | positivity               | Biomarker_Measurement |     0.8795   |
|  6 | Pan-CK                   | Biomarker             |     0.9975   |
|  7 | CK7                      | Biomarker             |     0.9975   |
|  8 | CK8/18                   | Biomarker             |     0.9987   |
|  9 | MUC1                     | Biomarker             |     0.9967   |
| 10 | MUC6                     | Biomarker             |     0.9972   |
| 11 | carbonic anhydrase IX    | Biomarker             |     0.937567 |
| 12 | CD10                     | Biomarker             |     0.9974   |
| 13 | EMA                      | Biomarker             |     0.9899   |
| 14 | β-catenin                | Biomarker             |     0.8059   |
| 15 | e-cadherin               | Biomarker             |     0.9806   |


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_biomarker|
|Compatibility:|Healthcare NLP 3.3.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


Trained on data sampled from Mimic-III, and annotated in-house.


## Benchmarking


```bash
label                      	 tp     fp    fn    prec        rec         f1        
I-Oncogenes                	 73     65    64    0.5289855   0.5328467   0.53090906
B-Radiotherapy             	 130    8     12    0.942029    0.91549295  0.9285714 
B-Chemotherapy             	 644    31    28    0.9540741   0.9583333   0.956199  
I-Radiotherapy             	 72     4     8     0.94736844  0.9         0.92307687
B-Predictive_Biomarkers    	 0      0     2     0.0         0.0         0.0       
I-Staging                  	 71     11    30    0.86585367  0.7029703   0.77595633
B-Radiological_Test_Result 	 0      3     20    0.0         0.0         0.0       
B-Drug                     	 18     10    19    0.64285713  0.4864865   0.5538461 
B-Dosage                   	 123    20    28    0.86013985  0.81456953  0.8367347 
I-Test_Result              	 22     11    44    0.6666667   0.33333334  0.44444448
I-CancerModifier           	 349    41    86    0.8948718   0.80229884  0.8460606 
I-Predictive_Biomarkers    	 0      0     1     0.0         0.0         0.0       
B-Date                     	 131    19    34    0.87333333  0.7939394   0.831746  
B-HormonalTherapy          	 114    5     12    0.9579832   0.9047619   0.9306123 
B-Radiological_Test        	 105    38    21    0.73426574  0.8333333   0.78066915
B-Ethnicity                	 8      0     1     1.0         0.8888889   0.94117653
I-Radiological_Test        	 69     50    15    0.57983196  0.8214286   0.67980295
I-UnspecificTherapy        	 59     8     6     0.880597    0.9076923   0.8939394 
I-Immunotherapy            	 100    25    22    0.8         0.8196721   0.80971664
B-UnspecificTherapy        	 92     16    12    0.8518519   0.88461536  0.8679245 
I-ResponseToTreatment      	 5      18    76    0.2173913   0.061728396 0.09615384
B-ResponseToTreatment      	 6      18    38    0.25        0.13636364  0.1764706 
B-Test_Result              	 23     17    20    0.575       0.53488374  0.55421686
I-Biomarker_Measurement    	 47     46    61    0.50537634  0.4351852   0.4676617 
B-Test                     	 286    145   138   0.6635731   0.6745283   0.6690058 
B-TargetedTherapy          	 675    74    75    0.9012016   0.9         0.9006004 
I-Biomarker                	 732    250   237   0.74541754  0.75541794  0.75038445
I-Radiological_Test_Result 	 8      6     86    0.5714286   0.08510638  0.14814815
B-CancerSurgery            	 194    29    34    0.8699552   0.85087717  0.86031044
I-Duration                 	 37     47    57    0.44047618  0.39361703  0.41573036
B-Oncogenes                	 342    118   229   0.74347824  0.5989492   0.66343355
I-CancerDx                 	 1272   131   123   0.90662867  0.911828    0.9092209 
I-Age                      	 19     4     4     0.82608694  0.82608694  0.826087  
B-Immunotherapy            	 300    29    16    0.9118541   0.9493671   0.9302325 
I-Prognostic_Biomarkers    	 4      3     7     0.5714286   0.36363637  0.44444445
B-Tumor_Finding            	 574    225   141   0.718398    0.8027972   0.75825626
B-CancerDx                 	 2620   205   169   0.9274336   0.9394048   0.9333808 
I-TargetedTherapy          	 317    70    38    0.8191214   0.89295775  0.8544474 
B-Gender                   	 52     14    10    0.7878788   0.83870965  0.81250006
B-Metastasis               	 584    41    44    0.9344      0.9299363   0.9321628 
I-Dosage                   	 69     16    19    0.8117647   0.78409094  0.7976879 
B-CancerModifier           	 852    135   166   0.8632219   0.83693516  0.84987533
B-Staging                  	 71     27    23    0.7244898   0.7553192   0.7395834 
I-Tumor_Finding            	 79     58    92    0.57664233  0.4619883   0.512987  
I-Test                     	 168    96    123   0.6363636   0.57731956  0.60540545
B-Age                      	 42     7     6     0.85714287  0.875       0.8659794 
I-HormonalTherapy          	 54     7     3     0.8852459   0.94736844  0.91525424
B-PerformanceStatus        	 11     2     0     0.84615386  1.0         0.9166667 
I-Chemotherapy             	 60     6     9     0.90909094  0.8695652   0.8888889 
I-Date                     	 116    15    9     0.8854962   0.928       0.90625   
B-Prognostic_Biomarkers    	 33     11    35    0.75        0.4852941   0.58928573
B-Duration                 	 30     50    38    0.375       0.44117647  0.40540543
I-Metastasis               	 32     14    45    0.6956522   0.41558442  0.5203252 
B-Biomarker_Measurement    	 437    124   175   0.7789661   0.71405226  0.745098  
I-CancerSurgery            	 128    17    30    0.8827586   0.8101266   0.8448845 
I-Drug                     	 2      0     8     1.0         0.2         0.3333333 
B-Biomarker                	 3027   571   332   0.8413007   0.9011611   0.8702027 
I-PerformanceStatus        	 37     15    0     0.71153843  1.0         0.83146065
Macro-average                15525  3026  3181  0.7223804   0.675604    0.69820964
Micro-average	             15525  3026  3181  0.8368821   0.8299476   0.8334004
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzI5ODA1NDM5XX0=
-->