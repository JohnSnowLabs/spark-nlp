---
layout: model
title: Detect Living Species (bert_embeddings_bert_base_fr_cased)
author: John Snow Labs
name: ner_living_species_bert
date: 2022-06-23
tags: [fr, ner, clinical, licensed, bert]
task: Named Entity Recognition
language: fr
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts in French which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture. This model is trained using `bert_embeddings_bert_base_fr_cased` embeddings.

It is trained on the [LivingNER](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/) corpus that is composed of clinical case reports extracted from miscellaneous medical specialties including COVID, oncology, infectious diseases, tropical medicine, urology, pediatrics, and others.

**NOTE :**
1.	The text files were translated from Spanish with a neural machine translation system.
2.	The annotations were translated with the same neural machine translation system.
3.	The translated annotations were transferred to the translated text files using an annotation transfer technology.

## Predicted Entities

`HUMAN`, `SPECIES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_bert_fr_3.5.3_3.0_1655973142130.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_fr_cased", "fr")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_living_species_bert", "fr", "clinical/models")\
.setInputCols(["sentence", "token", "embeddings"])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
document_assembler, 
sentence_detector,
tokenizer,
embeddings,
ner_model,
ner_converter   
])

data = spark.createDataFrame([["""Femme de 47 ans allergique à l'iode, fumeuse sociale, opérée pour des varices, deux césariennes et un abcès fessier. Vit avec son mari et ses trois enfants, travaille comme enseignante. Initialement, le patient a eu une bonne évolution, mais au 2ème jour postopératoire, il a commencé à montrer une instabilité hémodynamique. Les sérologies pour Coxiella burnetii, Bartonella henselae, Borrelia burgdorferi, Entamoeba histolytica, Toxoplasma gondii, herpès simplex virus 1 et 2, cytomégalovirus, virus d'Epstein Barr, virus de la varicelle et du zona et parvovirus B19 étaient négatives. Cependant, un test au rose Bengale positif pour Brucella, le test de Coombs et les agglutinations étaient également positifs avec un titre de 1/40."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_fr_cased", "fr")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_living_species_bert", "fr", "clinical/models")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")

val ner_converter = new NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, 
sentence_detector,
tokenizer,
embeddings,
ner_model,
ner_converter))

val data = Seq("""Femme de 47 ans allergique à l'iode, fumeuse sociale, opérée pour des varices, deux césariennes et un abcès fessier. Vit avec son mari et ses trois enfants, travaille comme enseignante. Initialement, le patient a eu une bonne évolution, mais au 2ème jour postopératoire, il a commencé à montrer une instabilité hémodynamique. Les sérologies pour Coxiella burnetii, Bartonella henselae, Borrelia burgdorferi, Entamoeba histolytica, Toxoplasma gondii, herpès simplex virus 1 et 2, cytomégalovirus, virus d'Epstein Barr, virus de la varicelle et du zona et parvovirus B19 étaient négatives. Cependant, un test au rose Bengale positif pour Brucella, le test de Coombs et les agglutinations étaient également positifs avec un titre de 1/40.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("fr.med_ner.living_species.bert").predict("""Femme de 47 ans allergique à l'iode, fumeuse sociale, opérée pour des varices, deux césariennes et un abcès fessier. Vit avec son mari et ses trois enfants, travaille comme enseignante. Initialement, le patient a eu une bonne évolution, mais au 2ème jour postopératoire, il a commencé à montrer une instabilité hémodynamique. Les sérologies pour Coxiella burnetii, Bartonella henselae, Borrelia burgdorferi, Entamoeba histolytica, Toxoplasma gondii, herpès simplex virus 1 et 2, cytomégalovirus, virus d'Epstein Barr, virus de la varicelle et du zona et parvovirus B19 étaient négatives. Cependant, un test au rose Bengale positif pour Brucella, le test de Coombs et les agglutinations étaient également positifs avec un titre de 1/40.""")
```

</div>

## Results

```bash
+--------------------------------+-------+
|ner_chunk                       |label  |
+--------------------------------+-------+
|Femme                           |HUMAN  |
|mari                            |HUMAN  |
|enfants                         |HUMAN  |
|patient                         |HUMAN  |
|Coxiella burnetii               |SPECIES|
|Bartonella henselae             |SPECIES|
|Borrelia burgdorferi            |SPECIES|
|Entamoeba histolytica           |SPECIES|
|Toxoplasma gondii               |SPECIES|
|cytomégalovirus                 |SPECIES|
|virus d'Epstein Barr            |SPECIES|
|virus de la varicelle et du zona|SPECIES|
|parvovirus B19                  |SPECIES|
|Brucella                        |SPECIES|
+--------------------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_living_species_bert|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|fr|
|Size:|16.4 MB|

## References

[https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/)

## Benchmarking

```bash
label         precision  recall  f1-score  support
B-HUMAN       0.81       0.95    0.87      2549    
B-SPECIES     0.66       0.87    0.75      2824    
I-HUMAN       0.98       0.43    0.60      114     
I-SPECIES     0.73       0.77    0.75      1109    
micro-avg     0.73       0.87    0.80      6596    
macro-avg     0.80       0.75    0.74      6596    
weighted-avg  0.74       0.87    0.80      6596  
```
