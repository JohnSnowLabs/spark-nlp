---
layout: model
title: Detect Living Species (bert_base_cased)
author: John Snow Labs
name: ner_living_species_bert
date: 2022-06-23
tags: [ro, ner, clinical, licensed, bert]
task: Named Entity Recognition
language: ro
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts in Romanian which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture. This model is trained using `bert_base_cased` embeddings.

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_bert_ro_3.5.3_3.0_1655974560466.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = BertEmbeddings.pretrained("bert_base_cased", "ro")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_living_species_bert", "ro", "clinical/models")\
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

data = spark.createDataFrame([["""O femeie în vârstă de 26 de ani, însărcinată în 11 săptămâni, a consultat serviciul de urgențe dermatologice pentru că prezenta, de 4 zile, leziuni punctiforme dureroase de debut brusc pe vârful degetelor. Pacientul raportează că leziunile au început pe degete și ulterior s-au extins la degetele de la picioare. Markerii de imunitate, ANA și crioagglutininele, au fost negativi, iar serologia VHB a indicat doar vaccinarea. Pe baza acestor rezultate, diagnosticul de vasculită a fost exclus și, având în vedere diagnosticul suspectat de erupție cutanată cu mănuși și șosete, s-a efectuat serologia pentru virusul Ebstein Barr. Exantemă la mănuși și șosete datorat parvovirozei B19. Având în vedere suspiciunea unei afecțiuni infecțioase cu aceste caracteristici, a fost solicitată serologia pentru EBV, enterovirus și parvovirus B19, cu IgM pozitiv pentru acesta din urmă în două ocazii. De asemenea, nu au existat semne de anemie fetală sau complicații ale acesteia."""]]).toDF("text")

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

val embeddings = BertEmbeddings.pretrained("bert_base_cased", "ro")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_living_species_bert", "ro", "clinical/models")
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

val data = Seq("""O femeie în vârstă de 26 de ani, însărcinată în 11 săptămâni, a consultat serviciul de urgențe dermatologice pentru că prezenta, de 4 zile, leziuni punctiforme dureroase de debut brusc pe vârful degetelor. Pacientul raportează că leziunile au început pe degete și ulterior s-au extins la degetele de la picioare. Markerii de imunitate, ANA și crioagglutininele, au fost negativi, iar serologia VHB a indicat doar vaccinarea. Pe baza acestor rezultate, diagnosticul de vasculită a fost exclus și, având în vedere diagnosticul suspectat de erupție cutanată cu mănuși și șosete, s-a efectuat serologia pentru virusul Ebstein Barr. Exantemă la mănuși și șosete datorat parvovirozei B19. Având în vedere suspiciunea unei afecțiuni infecțioase cu aceste caracteristici, a fost solicitată serologia pentru EBV, enterovirus și parvovirus B19, cu IgM pozitiv pentru acesta din urmă în două ocazii. De asemenea, nu au existat semne de anemie fetală sau complicații ale acesteia.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ro.med_ner.living_species.bert").predict("""O femeie în vârstă de 26 de ani, însărcinată în 11 săptămâni, a consultat serviciul de urgențe dermatologice pentru că prezenta, de 4 zile, leziuni punctiforme dureroase de debut brusc pe vârful degetelor. Pacientul raportează că leziunile au început pe degete și ulterior s-au extins la degetele de la picioare. Markerii de imunitate, ANA și crioagglutininele, au fost negativi, iar serologia VHB a indicat doar vaccinarea. Pe baza acestor rezultate, diagnosticul de vasculită a fost exclus și, având în vedere diagnosticul suspectat de erupție cutanată cu mănuși și șosete, s-a efectuat serologia pentru virusul Ebstein Barr. Exantemă la mănuși și șosete datorat parvovirozei B19. Având în vedere suspiciunea unei afecțiuni infecțioase cu aceste caracteristici, a fost solicitată serologia pentru EBV, enterovirus și parvovirus B19, cu IgM pozitiv pentru acesta din urmă în două ocazii. De asemenea, nu au existat semne de anemie fetală sau complicații ale acesteia.""")
```

</div>

## Results

```bash
+--------------------+-------+
|ner_chunk           |label  |
+--------------------+-------+
|femeie              |HUMAN  |
|Pacientul           |HUMAN  |
|VHB                 |SPECIES|
|virusul Ebstein Barr|SPECIES|
|parvovirozei B19    |SPECIES|
|EBV                 |SPECIES|
|enterovirus         |SPECIES|
|parvovirus B19      |SPECIES|
|fetală              |HUMAN  |
+--------------------+-------+
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
|Language:|ro|
|Size:|16.4 MB|

## References

[https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
B-HUMAN       0.85       0.94    0.89      2184    
B-SPECIES     0.75       0.85    0.80      2617    
I-HUMAN       0.89       0.11    0.20      72      
I-SPECIES     0.74       0.80    0.77      1027    
micro-avg     0.79       0.86    0.82      5900    
macro-avg     0.81       0.67    0.66      5900    
weighted-avg  0.79       0.86    0.82      5900   
```
