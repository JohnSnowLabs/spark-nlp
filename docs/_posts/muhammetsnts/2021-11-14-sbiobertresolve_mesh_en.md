---
layout: model
title: Sentence Entity Resolver for MeSH (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_mesh
date: 2021-11-14
tags: [mesh, entity_resolution, licensed, en, clinical]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.2
spark_version: 2.4
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities to Medical Subject Heading (MeSH) codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

## Predicted Entities

`MeSH Codes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_mesh_en_3.3.2_2.4_1636888521477.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

```sbiobertresolve_mesh``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_clinical``` as NER model. No need to set ```.setWhiteList()```.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
c2doc = Chunk2Doc()\
.setInputCols("ner_chunk")\
.setOutputCol("ner_chunk_doc") 

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sentence_embeddings")

mesh_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_mesh", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("mesh_code")\
.setDistanceFunction("EUCLIDEAN")\

resolver_pipeline = Pipeline(
stages = [
document_assembler,
sentenceDetectorDL,
tokenizer,
word_embeddings,
clinical_ner,
ner_converter,
c2doc,
sbert_embedder,
mesh_resolver
])


data = spark.createDataFrame([["""She was admitted to the hospital with chest pain and found to have bilateral pleural effusion, the right greater than the left. We reviewed the pathology obtained from the pericardectomy in March 2006, which was diagnostic of mesothelioma. At this time, chest tube placement for drainage of the fluid occurred and thoracoscopy with fluid biopsies, which were performed, which revealed malignant mesothelioma."""]]).toDF("text"))

result = resolver_pipeline.fit(data).transform(data)

```
```scala
...

val c2doc = Chunk2Doc()
.setInputCols("ner_chunk")
.setOutputCol("ner_chunk_doc") 

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sentence_embeddings")

val mesh_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_mesh", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sentence_embeddings"))
.setOutputCol("mesh_code")
.setDistanceFunction("EUCLIDEAN")

val resolver_pipeline = new Pipeline(
stages = Array(
document_assembler,
sentenceDetectorDL,
tokenizer,
word_embeddings,
clinical_ner,
ner_converter,
c2doc,
sbert_embedder,
mesh_resolver))

val data = Seq("She was admitted to the hospital with chest pain and found to have bilateral pleural effusion, the right greater than the left. We reviewed the pathology obtained from the pericardectomy in March 2006, which was diagnostic of mesothelioma. At this time, chest tube placement for drainage of the fluid occurred and thoracoscopy with fluid biopsies, which were performed, which revealed malignant mesothelioma.")

val result = resolver_pipeline.fit.(data).transform(data)

```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.mesh").predict("""She was admitted to the hospital with chest pain and found to have bilateral pleural effusion, the right greater than the left. We reviewed the pathology obtained from the pericardectomy in March 2006, which was diagnostic of mesothelioma. At this time, chest tube placement for drainage of the fluid occurred and thoracoscopy with fluid biopsies, which were performed, which revealed malignant mesothelioma.""")
```

</div>

## Results

```bash
+--------------------------+---------+----------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
|                 ner_chunk|   entity| mesh_code|                                                                                           all_codes|                                                                                         resolutions|                                                                                           distances|
+--------------------------+---------+----------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
|                chest pain|  PROBLEM|   D002637|D002637:::D059350:::D019547:::D020069:::D015746:::D000072716:::D005157:::D059265:::D001416:::D048...|Chest Pain:::Chronic Pain:::Neck Pain:::Shoulder Pain:::Abdominal Pain:::Cancer Pain:::Facial Pai...|0.0000:::0.0577:::0.0587:::0.0601:::0.0658:::0.0704:::0.0712:::0.0741:::0.0766:::0.0778:::0.0794:...|
|bilateral pleural effusion|  PROBLEM|   D010996|D010996:::D010490:::D011654:::D016724:::D010995:::D016066:::D011001:::D007819:::D035422:::D004653...|Pleural Effusion:::Pericardial Effusion:::Pulmonary Edema:::Empyema, Pleural:::Pleural Diseases::...|0.0309:::0.1010:::0.1115:::0.1213:::0.1218:::0.1398:::0.1425:::0.1401:::0.1451:::0.1464:::0.1464:...|
|             the pathology|     TEST|   D010336|D010336:::D010335:::D001004:::D020969:::C001675:::C536472:::D004194:::D003951:::D013631:::C535329...|Pathology:::Pathologic Processes:::Anus Diseases:::Disease Attributes:::malformins:::Upington dis...|0.0788:::0.0977:::0.1364:::0.1396:::0.1419:::0.1459:::0.1418:::0.1393:::0.1514:::0.1541:::0.1491:...|
|        the pericardectomy|TREATMENT|   D010492|D010492:::D011670:::D018700:::D020884:::D011672:::D005927:::D064727:::D002431:::C000678968:::D011...|Pericardiectomy:::Pulpectomy:::Pleurodesis:::Colpotomy:::Pulpotomy:::Glossectomy:::Posterior Caps...|0.1098:::0.1448:::0.1801:::0.1852:::0.1871:::0.1923:::0.1901:::0.2023:::0.2075:::0.2010:::0.1996:...|
|              mesothelioma|  PROBLEM|D000086002|D000086002:::C535700:::D009208:::D032902:::D018301:::D018199:::C562740:::C000686536:::D018276:::D...|Mesothelioma, Malignant:::Malignant mesenchymal tumor:::Myoepithelioma:::Ganoderma:::Neoplasms, M...|0.0813:::0.1515:::0.1599:::0.1810:::0.1864:::0.1881:::0.1907:::0.1938:::0.1924:::0.1876:::0.2040:...|
|      chest tube placement|TREATMENT|   D015505|D015505:::D019616:::D013896:::D012124:::D013906:::D013510:::D020708:::D035423:::D013903:::D000066...|Chest Tubes:::Thoracic Surgical Procedures:::Thoracic Diseases:::Respiratory Care Units:::Thoraco...|0.0557:::0.1473:::0.1598:::0.1604:::0.1725:::0.1651:::0.1795:::0.1760:::0.1804:::0.1846:::0.1883:...|
|     drainage of the fluid|TREATMENT|   D004322|D004322:::D018495:::C045413:::D021061:::D045268:::D018508:::D005441:::D015633:::D014906:::D001834...|Drainage:::Fluid Shifts:::Bonain's liquid:::Liquid Ventilation:::Flowmeters:::Water Purification:...|0.1141:::0.1403:::0.1582:::0.1549:::0.1586:::0.1626:::0.1599:::0.1655:::0.1667:::0.1656:::0.1741:...|
|              thoracoscopy|TREATMENT|   D013906|D013906:::D020708:::D035423:::D013905:::D035441:::D013897:::D001468:::D000069258:::D013909:::D013...|Thoracoscopy:::Thoracoscopes:::Thoracic Cavity:::Thoracoplasty:::Thoracic Wall:::Thoracic Duct:::...|0.0000:::0.0359:::0.0744:::0.1007:::0.1070:::0.1143:::0.1186:::0.1257:::0.1228:::0.1356:::0.1354:...|
|            fluid biopsies|     TEST|D000073890|D000073890:::D010533:::D020420:::D011677:::D017817:::D001706:::D005441:::D005751:::D013582:::D000...|Liquid Biopsy:::Peritoneal Lavage:::Cyst Fluid:::Punctures:::Nasal Lavage Fluid:::Biopsy:::Fluids...|0.1408:::0.1612:::0.1763:::0.1744:::0.1744:::0.1810:::0.1744:::0.1828:::0.1896:::0.1909:::0.1950:...|
|    malignant mesothelioma|  PROBLEM|D000086002|D000086002:::C535700:::C562740:::D009236:::D007890:::D012515:::D009208:::C009823:::C000683999:::C...|Mesothelioma, Malignant:::Malignant mesenchymal tumor:::Hemangiopericytoma, Malignant:::Myxosarco...|0.0737:::0.1106:::0.1658:::0.1627:::0.1660:::0.1639:::0.1728:::0.1676:::0.1791:::0.1843:::0.1849:...|
+-------+--------------------------+---------+----------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_mesh|
|Compatibility:|Healthcare NLP 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[mesh_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on 01 December 2021 MeSH dataset.
