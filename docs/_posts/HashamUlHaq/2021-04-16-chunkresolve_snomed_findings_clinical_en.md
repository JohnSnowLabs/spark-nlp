---
layout: model
title: SNOMED ChunkResolver
author: John Snow Labs
name: chunkresolve_snomed_findings_clinical
date: 2021-04-16
tags: [entity_resolution, clinical, licensed, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
deprecated: true
annotator: ChunkEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.

## Predicted Entities

Snomed Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_snomed_findings_clinical_en_3.0.0_3.0_1618603404974.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

snomed_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_snomed_findings_clinical","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("snomed_resolution")

pipeline_snomed = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, snomed_ner_converter, chunk_embeddings, snomed_resolver])

data = ["""Pentamidine 300 mg IV q . 36 hours , Pentamidine nasal wash 60 mg per 6 ml of sterile water q.d . , voriconazole 200 mg p.o . b.i.d . , acyclovir 400 mg p.o . b.i.d . , cyclosporine 50 mg p.o . b.i.d . , prednisone 60 mg p.o . q.d . , GCSF 480 mcg IV q.d . , Epogen 40,000 units subcu q . week , Protonix 40 mg q.d . , Simethicone 80 mg p.o . q . 8 , nitroglycerin paste 1 " ; q . 4 h . p.r.n . , flunisolide nasal inhaler , 2 puffs q . 8 , OxyCodone 10-15 mg p.o . q . 6 p.r.n . , Sudafed 30 mg q . 6 p.o . p.r.n . , Fluconazole 2% cream b.i.d . to erythematous skin lesions , Ditropan 5 mg p.o . b.i.d . , Tylenol 650 mg p.o . q . 4 h . p.r.n . , Ambien 5-10 mg p.o . q . h.s . p.r.n . , Neurontin 100 mg q . a.m . , 200 mg q . p.m . , Aquaphor cream b.i.d . p.r.n . , Lotrimin 1% cream b.i.d . to feet , Dulcolax 5-10 mg p.o . q.d . p.r.n . , Phoslo 667 mg p.o . t.i.d . , Peridex 0.12% , 15 ml p.o . b.i.d . mouthwash , Benadryl 25-50 mg q . 4-6 h . p.r.n . pruritus , Sarna cream q.d . p.r.n . pruritus , Nystatin 5 ml p.o . q.i.d . swish and !""",

"""Albuterol nebulizers 2.5 mg q.4h . and Atrovent nebulizers 0.5 mg q.4h . , please alternate albuterol and Atrovent ; Rocaltrol 0.25 mcg per NG tube q.d .; calcium carbonate 1250 mg per NG tube q.i.d .; vitamin B12 1000 mcg IM q . month , next dose is due Nov 18 ; diltiazem 60 mg per NG tube t.i.d .; ferrous sulfate 300 mg per NG t.i.d .; Haldol 5 mg IV q.h.s .; hydralazine 10 mg IV q.6h . p.r.n . hypertension ; lisinopril 10 mg per NG tube q.d .; Ativan 1 mg per NG tube q.h.s .; Lopressor 25 mg per NG tube t.i.d .; Zantac 150 mg per NG tube b.i.d .; multivitamin 10 ml per NG tube q.d .; Macrodantin 100 mg per NG tube q.i.d . x 10 days beginning on 11/3/00 .""",

"""Tylenol 650 mg p.o . q . 4-6h p.r.n . headache or pain ; acyclovir 400 mg p.o . t.i.d .; acyclovir topical t.i.d . to be applied to lesion on corner of mouth ; Peridex 15 ml p.o . b.i.d .; Mycelex 1 troche p.o . t.i.d .; g-csf 404 mcg subcu q.d .; folic acid 1 mg p.o . q.d .; lorazepam 1-2 mg p.o . q . 4-6h p.r.n . nausea and vomiting ; Miracle Cream topical q.d . p.r.n . perianal irritation ; Eucerin Cream topical b.i.d .; Zantac 150 mg p.o . b.i.d .; Restoril 15-30 mg p.o . q . h.s . p.r.n . insomnia ; multivitamin 1 tablet p.o . q.d .; viscous lidocaine 15 ml p.o . q . 3h can be applied to corner of mouth or lips p.r.n . pain control ."""]

model = pipeline_snomed.fit(spark.createDataFrame([['']]).toDF("text"))

results = model.transform(spark.createDataFrame([["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."]], ["text"]))
```
```scala
...

val snomed_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_snomed_findings_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("snomed_resolution")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, snomed_ner_converter, chunk_embeddings, snomed_resolver))

val data = Array("""Pentamidine 300 mg IV q . 36 hours , Pentamidine nasal wash 60 mg per 6 ml of sterile water q.d . , voriconazole 200 mg p.o . b.i.d . , acyclovir 400 mg p.o . b.i.d . , cyclosporine 50 mg p.o . b.i.d . , prednisone 60 mg p.o . q.d . , GCSF 480 mcg IV q.d . , Epogen 40,000 units subcu q . week , Protonix 40 mg q.d . , Simethicone 80 mg p.o . q . 8 , nitroglycerin paste 1 " ; q . 4 h . p.r.n . , flunisolide nasal inhaler , 2 puffs q . 8 , OxyCodone 10-15 mg p.o . q . 6 p.r.n . , Sudafed 30 mg q . 6 p.o . p.r.n . , Fluconazole 2% cream b.i.d . to erythematous skin lesions , Ditropan 5 mg p.o . b.i.d . , Tylenol 650 mg p.o . q . 4 h . p.r.n . , Ambien 5-10 mg p.o . q . h.s . p.r.n . , Neurontin 100 mg q . a.m . , 200 mg q . p.m . , Aquaphor cream b.i.d . p.r.n . , Lotrimin 1% cream b.i.d . to feet , Dulcolax 5-10 mg p.o . q.d . p.r.n . , Phoslo 667 mg p.o . t.i.d . , Peridex 0.12% , 15 ml p.o . b.i.d . mouthwash , Benadryl 25-50 mg q . 4-6 h . p.r.n . pruritus , Sarna cream q.d . p.r.n . pruritus , Nystatin 5 ml p.o . q.i.d . swish and !""",

"""Albuterol nebulizers 2.5 mg q.4h . and Atrovent nebulizers 0.5 mg q.4h . , please alternate albuterol and Atrovent ; Rocaltrol 0.25 mcg per NG tube q.d .; calcium carbonate 1250 mg per NG tube q.i.d .; vitamin B12 1000 mcg IM q . month , next dose is due Nov 18 ; diltiazem 60 mg per NG tube t.i.d .; ferrous sulfate 300 mg per NG t.i.d .; Haldol 5 mg IV q.h.s .; hydralazine 10 mg IV q.6h . p.r.n . hypertension ; lisinopril 10 mg per NG tube q.d .; Ativan 1 mg per NG tube q.h.s .; Lopressor 25 mg per NG tube t.i.d .; Zantac 150 mg per NG tube b.i.d .; multivitamin 10 ml per NG tube q.d .; Macrodantin 100 mg per NG tube q.i.d . x 10 days beginning on 11/3/00 .""",

"""Tylenol 650 mg p.o . q . 4-6h p.r.n . headache or pain ; acyclovir 400 mg p.o . t.i.d .; acyclovir topical t.i.d . to be applied to lesion on corner of mouth ; Peridex 15 ml p.o . b.i.d .; Mycelex 1 troche p.o . t.i.d .; g-csf 404 mcg subcu q.d .; folic acid 1 mg p.o . q.d .; lorazepam 1-2 mg p.o . q . 4-6h p.r.n . nausea and vomiting ; Miracle Cream topical q.d . p.r.n . perianal irritation ; Eucerin Cream topical b.i.d .; Zantac 150 mg p.o . b.i.d .; Restoril 15-30 mg p.o . q . h.s . p.r.n . insomnia ; multivitamin 1 tablet p.o . q.d .; viscous lidocaine 15 ml p.o . q . 3h can be applied to corner of mouth or lips p.r.n . pain control .""")

val result = pipeline.fit(Seq.empty[String]).transform(data)
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------+-------+----------------------------------------------------------------------------------------------------+-----------------+----------+
|                                                                        chunk| entity|                                                                                         target_text|             code|confidence|
+-----------------------------------------------------------------------------+-------+----------------------------------------------------------------------------------------------------+-----------------+----------+
|                                                    erythematous skin lesions|PROBLEM|Skin lesion:::Achromic skin lesions of pinta:::Scaly skin:::Skin constricture:::Cratered skin les...|         95324001|    0.0937|
|                                                                     pruritus|PROBLEM|Pruritus:::Genital pruritus:::Postmenopausal pruritus:::Pruritus hiemalis:::Pruritus ani:::Anogen...|        418363000|    0.1394|
|                                                                     pruritus|PROBLEM|Pruritus:::Genital pruritus:::Postmenopausal pruritus:::Pruritus hiemalis:::Pruritus ani:::Anogen...|        418363000|    0.1394|
|                                                                 hypertension|PROBLEM|Hypertension:::Renovascular hypertension:::Idiopathic hypertension:::Venous hypertension:::Resist...|         38341003|    0.1019|
|                                                             headache or pain|PROBLEM|Pain:::Headache:::Postchordotomy pain:::Throbbing pain:::Aching headache:::Postspinal headache:::...|         22253000|    0.0953|
|                                         applied to lesion on corner of mouth|PROBLEM|Lesion of tongue:::Erythroleukoplakia of mouth:::Lesion of nose:::Lesion of oropharynx:::Erythrop...|        300246005|    0.0547|
|                                                          nausea and vomiting|PROBLEM|Nausea and vomiting:::Vomiting without nausea:::Nausea:::Intractable nausea and vomiting:::Vomiti...|         16932000|    0.0995|
|                                                          perianal irritation|PROBLEM|Perineal irritation:::Vulval irritation:::Skin irritation:::Perianal pain:::Perianal itch:::Vagin...|        281639001|    0.0764|
|                                                                     insomnia|PROBLEM|Insomnia:::Mood insomnia:::Nonorganic insomnia:::Persistent insomnia:::Psychophysiologic insomnia...|        193462001|    0.1198|
+-----------------------------------------------------------------------------+-------+----------------------------------------------------------------------------------------------------+-----------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_snomed_findings_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[icd10cm]|
|Language:|en|
