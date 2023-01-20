---
layout: model
title: Extract neurologic deficits related to Stroke Scale (NIHSS)
author: John Snow Labs
name: ner_nihss
date: 2021-11-15
tags: [ner, en, licensed, clinical]
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


The National Institutes of Health Stroke Scale (NIHSS) is a 15-item neurologic examination stroke scale. It quantifies the physical manifestations of neurological deficits and provides crucial support for clinical decision making and early-stage emergency triage.


## Predicted Entities


`11_ExtinctionInattention`, `6b_RightLeg`, `1c_LOCCommands`, `10_Dysarthria`, `NIHSS`, `5_Motor`, `8_Sensory`, `4_FacialPalsy`, `6_Motor`, `2_BestGaze`, `Measurement`, `6a_LeftLeg`, `5b_RightArm`, `5a_LeftArm`, `1b_LOCQuestions`, `3_Visual`, `9_BestLanguage`, `7_LimbAtaxia`, `1a_LOC`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_NIHSS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_nihss_en_3.3.2_3.0_1636997459858.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_nihss_en_3.3.2_3.0_1636997459858.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") \
	.setInputCols(["document"]) \
	.setOutputCol("sentence")

tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")
 
embeddings_clinical = WordEmbeddingsModel.pretrained('embeddings_clinical', 'en', 'clinical/models') \
    .setInputCols(['sentence', 'token']) \
    .setOutputCol('embeddings')

clinical_ner = MedicalNerModel.pretrained("ner_nihss", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("entities")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["Abdomen , soft , nontender . NIH stroke scale on presentation was 23 to 24 for , one for consciousness , two for month and year and two for eye / grip , one to two for gaze , two for face , eight for motor , one for limited ataxia , one to two for sensory , three for best language and two for attention . On the neurologic examination the patient was intermittently"]], ["text"]))
```
```scala

val document_assembler = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
	.setInputCols("document")
	.setOutputCol("sentence")

val tokenizer = new Tokenizer()
	.setInputCols("sentence")
	.setOutputCol("token")

val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_nihss", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val data = Seq("""Abdomen , soft , nontender . NIH stroke scale on presentation was 23 to 24 for , one for consciousness , two for month and year and two for eye / grip , one to two for gaze , two for face , eight for motor , one for limited ataxia , one to two for sensory , three for best language and two for attention . On the neurologic examination the patient was intermittently""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}

```python
import nlu
nlu.load("en.med_ner.nihss").predict("""Abdomen , soft , nontender . NIH stroke scale on presentation was 23 to 24 for , one for consciousness , two for month and year and two for eye / grip , one to two for gaze , two for face , eight for motor , one for limited ataxia , one to two for sensory , three for best language and two for attention . On the neurologic examination the patient was intermittently""")
```

</div>


## Results


```bash
|    | chunk              | entity                   |
|---:|:-------------------|:-------------------------|
|  0 | NIH stroke scale   | NIHSS                    |
|  1 | 23 to 24           | Measurement              |
|  2 | one                | Measurement              |
|  3 | consciousness      | 1a_LOC                   |
|  4 | two                | Measurement              |
|  5 | month and year and | 1b_LOCQuestions          |
|  6 | two                | Measurement              |
|  7 | eye / grip         | 1c_LOCCommands           |
|  8 | one to             | Measurement              |
|  9 | two                | Measurement              |
| 10 | gaze               | 2_BestGaze               |
| 11 | two                | Measurement              |
| 12 | face               | 4_FacialPalsy            |
| 13 | eight              | Measurement              |
| 14 | one                | Measurement              |
| 15 | limited            | 7_LimbAtaxia             |
| 16 | ataxia             | 7_LimbAtaxia             |
| 17 | one to two         | Measurement              |
| 18 | sensory            | 8_Sensory                |
| 19 | three              | Measurement              |
| 20 | best language      | 9_BestLanguage           |
| 21 | two                | Measurement              |
| 22 | attention          | 11_ExtinctionInattention |


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_nihss|
|Compatibility:|Healthcare NLP 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


@article{wangnational,
title={National Institutes of Health Stroke Scale (NIHSS) Annotations for the MIMIC-III Database},
author={Wang, Jiayang and Huang, Xiaoshuo and Yang, Lin and Li, Jiao}
}


## Benchmarking


```bash
label                      	 tp  	 fp 	 fn 	 prec       	 rec        	 f1         
I-NIHSS                    	 126 	 5  	 14 	 0.96183205 	 0.9        	 0.92988926 
B-NIHSS                    	 152 	 9  	 14 	 0.94409937 	 0.91566265 	 0.9296636  
B-5b_RightArm              	 33  	 0  	 7  	 1.0        	 0.825      	 0.90410954 
I-Measurement              	 17  	 1  	 69 	 0.9444444  	 0.19767442 	 0.3269231  
I-5_Motor                  	 12  	 3  	 2  	 0.8        	 0.85714287 	 0.82758623 
I-1a_LOC                   	 134 	 1  	 3  	 0.9925926  	 0.9781022  	 0.9852941  
I-9_BestLanguage           	 85  	 3  	 0  	 0.96590906 	 1.0        	 0.982659   
B-7_LimbAtaxia             	 39  	 0  	 3  	 1.0        	 0.9285714  	 0.9629629  
B-4_FacialPalsy            	 53  	 4  	 4  	 0.9298246  	 0.9298246  	 0.9298246  
B-1a_LOC                   	 39  	 0  	 5  	 1.0        	 0.8863636  	 0.939759   
B-6a_LeftLeg               	 35  	 0  	 4  	 1.0        	 0.8974359  	 0.945946   
B-10_Dysarthria            	 51  	 2  	 4  	 0.9622642  	 0.92727274 	 0.94444454 
B-8_Sensory                	 43  	 3  	 7  	 0.9347826  	 0.86       	 0.8958333  
I-6b_RightLeg              	 149 	 5  	 0  	 0.96753246 	 1.0        	 0.9834984  
B-3_Visual                 	 40  	 0  	 6  	 1.0        	 0.8695652  	 0.9302325  
B-5_Motor                  	 4   	 1  	 4  	 0.8        	 0.5        	 0.61538464 
I-6a_LeftLeg               	 157 	 1  	 6  	 0.9936709  	 0.9631902  	 0.9781932  
B-5a_LeftArm               	 37  	 1  	 4  	 0.9736842  	 0.902439   	 0.93670887 
I-4_FacialPalsy            	 129 	 5  	 4  	 0.96268654 	 0.9699248  	 0.96629214 
B-2_BestGaze               	 41  	 1  	 2  	 0.97619045 	 0.95348835 	 0.9647058  
I-8_Sensory                	 78  	 0  	 10 	 1.0        	 0.8863636  	 0.939759   
I-5b_RightArm              	 153 	 1  	 12 	 0.9935065  	 0.92727274 	 0.95924765 
B-9_BestLanguage           	 45  	 1  	 5  	 0.9782609  	 0.9        	 0.9375     
I-5a_LeftArm               	 159 	 3  	 10 	 0.9814815  	 0.9408284  	 0.96072507 
I-1c_LOCCommands           	 109 	 1  	 0  	 0.9909091  	 1.0        	 0.9954338  
I-6_Motor                  	 12  	 4  	 4  	 0.75       	 0.75       	 0.75       
B-1b_LOCQuestions          	 43  	 0  	 2  	 1.0        	 0.95555556 	 0.97727275 
B-6b_RightLeg              	 32  	 1  	 2  	 0.969697   	 0.9411765  	 0.9552239  
I-2_BestGaze               	 112 	 0  	 2  	 1.0        	 0.98245615 	 0.99115044 
I-7_LimbAtaxia             	 113 	 1  	 0  	 0.99122804 	 1.0        	 0.9955947  
I-11_ExtinctionInattention 	 142 	 0  	 7  	 1.0        	 0.95302016 	 0.97594506 
B-1c_LOCCommands           	 40  	 0  	 1  	 1.0        	 0.9756098  	 0.9876543  
B-6_Motor                  	 4   	 1  	 5  	 0.8        	 0.44444445 	 0.57142854 
I-3_Visual                 	 103 	 1  	 5  	 0.99038464 	 0.9537037  	 0.9716981  
B-11_ExtinctionInattention 	 44  	 1  	 3  	 0.9777778  	 0.9361702  	 0.9565217  
B-Measurement              	 787 	 23 	 13 	 0.97160494 	 0.98375    	 0.97763973 
I-10_Dysarthria            	 76  	 0  	 1  	 1.0        	 0.987013   	 0.99346405 
I-1b_LOCQuestions          	 114 	 0  	 6  	 1.0        	 0.95       	 0.9743589  
Macro-average	             3542    83      250     0.9606412       0.8876058       0.9226804
Micro-average	             3542    83      250     0.9771035       0.9340717       0.9551032
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDAyMzUyODUwXX0=
-->