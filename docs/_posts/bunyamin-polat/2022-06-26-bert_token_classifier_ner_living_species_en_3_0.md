---
layout: model
title: Detect Living Species
author: John Snow Labs
name: bert_token_classifier_ner_living_species
date: 2022-06-26
tags: [en, ner, clinical, licensed, bertfortokenclassification]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture. This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP.

It is trained on the [LivingNER](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/) corpus that is composed of clinical case reports extracted from miscellaneous medical specialties including COVID, oncology, infectious diseases, tropical medicine, urology, pediatrics, and others.

**NOTE :**
- The text files were translated from Spanish with a neural machine translation system.
- The annotations were translated with the same neural machine translation system.
- The translated annotations were transferred to the translated text files using an annotation transfer technology.

## Predicted Entities

`HUMAN`, `SPECIES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_living_species_en_3.5.3_3.0_1656273939035.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_living_species_en_3.5.3_3.0_1656273939035.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")\

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_living_species", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("ner")\
.setCaseSensitive(True)\
.setMaxSentenceLength(512)

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
document_assembler, 
sentence_detector,
tokenizer,
ner_model,
ner_converter   
])

data = spark.createDataFrame([["""42-year-old woman with end-stage chronic kidney disease, secondary to lupus nephropathy, and on peritoneal dialysis. History of four episodes of bacterial peritonitis and change of Tenckhoff catheter six months prior to admission due to catheter dysfunction. Three peritoneal fluid samples during her hospitalisation tested positive for Fusarium spp. The patient responded favourably and continued outpatient treatment with voriconazole (4mg/kg every 12 hours orally). All three isolates were identified as species of the Fusarium solani complex. In vitro susceptibility to itraconazole, voriconazole and posaconazole, according to Clinical and Laboratory Standards Institute - CLSI (M38-A) methodology, showed a minimum inhibitory concentration (MIC) in all three isolates and for all three antifungals of >16 μg/mL."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_living_species", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("ner")
.setCaseSensitive(True)
.setMaxSentenceLength(512)

val ner_converter = new NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(
document_assembler, 
sentence_detector,
tokenizer,
ner_model,
ner_converter))

val data = Seq("""42-year-old woman with end-stage chronic kidney disease, secondary to lupus nephropathy, and on peritoneal dialysis. History of four episodes of bacterial peritonitis and change of Tenckhoff catheter six months prior to admission due to catheter dysfunction. Three peritoneal fluid samples during her hospitalisation tested positive for Fusarium spp. The patient responded favourably and continued outpatient treatment with voriconazole (4mg/kg every 12 hours orally). All three isolates were identified as species of the Fusarium solani complex. In vitro susceptibility to itraconazole, voriconazole and posaconazole, according to Clinical and Laboratory Standards Institute - CLSI (M38-A) methodology, showed a minimum inhibitory concentration (MIC) in all three isolates and for all three antifungals of >16 μg/mL.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.living_species.token_bert").predict("""42-year-old woman with end-stage chronic kidney disease, secondary to lupus nephropathy, and on peritoneal dialysis. History of four episodes of bacterial peritonitis and change of Tenckhoff catheter six months prior to admission due to catheter dysfunction. Three peritoneal fluid samples during her hospitalisation tested positive for Fusarium spp. The patient responded favourably and continued outpatient treatment with voriconazole (4mg/kg every 12 hours orally). All three isolates were identified as species of the Fusarium solani complex. In vitro susceptibility to itraconazole, voriconazole and posaconazole, according to Clinical and Laboratory Standards Institute - CLSI (M38-A) methodology, showed a minimum inhibitory concentration (MIC) in all three isolates and for all three antifungals of >16 μg/mL.""")
```

</div>

## Results

```bash
+-----------------------+-------+
|ner_chunk              |label  |
+-----------------------+-------+
|woman                  |HUMAN  |
|bacterial              |SPECIES|
|Fusarium spp           |SPECIES|
|patient                |HUMAN  |
|species                |SPECIES|
|Fusarium solani complex|SPECIES|
|antifungals            |SPECIES|
+-----------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_living_species|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

[https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
B-HUMAN       0.83       0.96    0.89      2950    
B-SPECIES     0.70       0.93    0.80      3129    
I-HUMAN       0.73       0.39    0.51      145     
I-SPECIES     0.67       0.81    0.74      1166    
micro-avg     0.74       0.91    0.82      7390    
macro-avg     0.73       0.77    0.73      7390    
weighted-avg  0.75       0.91    0.82      7390  
```
