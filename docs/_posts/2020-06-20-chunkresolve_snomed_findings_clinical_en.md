---
layout: model
title: SNOMED ChunkResolver
author: John Snow Labs
name: chunkresolve_snomed_findings_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-06-20
tags: [clinical,entity_resolution,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.

## Predicted Entities 
Snomed Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_snomed_findings_clinical_en_2.5.1_2.4_1592617161564.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...

snomed_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_snomed_findings_clinical","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("snomed_resolution")
    
pipeline_snomed = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, snomed_ner_converter, chunk_embeddings, snomed_resolver])

model = pipeline_snomed.fit(spark.createDataFrame([['']]).toDF("text"))

results = model.transform(data)
```

```scala
...

val snomed_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_snomed_findings_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("snomed_resolution")
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, snomed_ner_converter, chunk_embeddings, snomed_resolver))

val result = pipeline.fit(Seq.empty[''].toDS.toDF("text")).transform(data)    

```
</div>

{:.h2_title}
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
|                                    patient's incisions sternal and right leg|PROBLEM|Laceration of right lower leg:::Varicose veins of right leg:::Stab wound of right lower leg:::Clo...|10963831000119103|    0.0526|
|                                                               blood pressure|   TEST|Elevated blood pressure:::Abnormal blood pressure:::High blood pressure:::Low blood pressure:::No...|         24184005|    0.0480|
|                                                                   hematocrit|   TEST|Hematocrit:::Stable hematocrit:::Precipitous drop in hematocrit:::Hematocrit - borderline high:::...|        365616005|    0.5722|
|                                                           bun and creatinine|   TEST|Micropunctum lacrimale:::Creatinine in sample:::Serum creatinine normal:::Serum creatinine low:::...|         95505003|    0.1501|
|                                                       prothrombin time level|   TEST|Prothrombin time low:::Prothrombin time increased:::Prothrombin time finding:::Prothrombin time n...|        165569003|    0.1356|
|                                                                  chest x-ray|   TEST|Flat chest:::Rigid chest:::Chest hyperinflated:::Chest percussion tympanitic:::Chest mass:::Chest...|          3274008|    0.0560|
|small bilateral effusions with mild cardiomegaly and subsegmental atelectasis|PROBLEM|Bilateral pleural effusion (disorder):::Effusion of joint of bilateral ankles:::Effusion of joint...|        425802001|    0.0501|
|                                              bibasilar and electrocardiogram|   TEST|Electrocardiogram abnormal:::Electrocardiogram normal:::Electrocardiogram finding:::Electrocardio...|        102594003|    0.0703|
|                                  acute ischemic changes on electrocardiogram|PROBLEM|Postoperative electrocardiogram changes:::Acute ischemic renal failure:::Potential for acute isch...|        251144001|    0.1238|
|                                                                 hypertension|PROBLEM|Hypertension:::Renovascular hypertension:::Idiopathic hypertension:::Venous hypertension:::Resist...|         38341003|    0.1019|
|                                                  chronic renal insufficiency|PROBLEM|Chronic progressive renal insufficiency:::Renal insufficiency:::Chronic insufficiency:::Anemia of...|        425369003|    0.1171|
+-----------------------------------------------------------------------------+-------+----------------------------------------------------------------------------------------------------+-----------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|---------------------------------------|
| Name:           | chunkresolve_snomed_findings_clinical |
| Type:    | ChunkEntityResolverModel              |
| Compatibility:  | Spark NLP 2.5.1+                                 |
| License:        | Licensed                              |
|Edition:|Official|                            |
|Input labels:         | [token, chunk_embeddings ]              |
|Output labels:        | [entity]                                |
| Language:       | en                                    |
| Case sensitive: | True                                  |
| Dependencies:  | embeddings_clinical                   |

{:.h2_title}
## Data Source
Trained on SNOMED CT Findings
http://www.snomed.org/
