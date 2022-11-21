---
layout: model
title: Sentence Entity Resolver for NDC (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_ndc
date: 2021-11-27
tags: [ndc, entity_resolution, licensed, en, cilnical]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.2
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model maps clinical entities and concepts (like drugs/ingredients) to [National Drug Codes](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory) using `sbiobert_base_cased_mli` Sentence Bert Embeddings. Also, if a drug has more than one NDC code, it returns all other codes in the all_k_aux_label column separated by `|` symbol.


## Predicted Entities


`NDC Codes`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_NDC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_ndc_en_3.3.2_2.4_1638010818380.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use


```sbiobertresolve_ndc``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_posology_greedy``` as NER model. ```DRUG``` set in ```.setWhiteList()```.


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
c2doc = Chunk2Doc()\
.setInputCols("ner_chunk")\
.setOutputCol("ner_chunk_doc") 

sbert_embedder = BertSentenceEmbeddings\
.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sentence_embeddings")

ndc_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_ndc", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("ndc_code")\
.setDistanceFunction("EUCLIDEAN")\
.setCaseSensitive(False)

resolver_pipeline = Pipeline(
stages = [
document_assembler,
sentenceDetectorDL,
tokenizer,
word_embeddings,
posology_ner,
ner_converter_icd,
c2doc,
sbert_embedder,
ndc_resolver
])

data = spark.createDataFrame([["""The patient was transferred secondary to inability and continue of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given aspirin 81 mg, folic acid 1 g daily, insulin glargine 100 UNT/ML injection and metformin 500 mg p.o. p.r.n."""]]).toDF("text")

result = resolver_pipeline.fit(data).transform(data)
```
```scala
...
val c2doc = new Chunk2Doc()
.setInputCols("ner_chunk")
.setOutputCol("ner_chunk_doc") 

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sentence_embeddings")

val ndc_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_ndc", "en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sentence_embeddings")) 
.setOutputCol("ndc_code")
.setDistanceFunction("EUCLIDEAN")
.setCaseSensitive(False)

val resolver_pipeline = new Pipeline().setStages(Array(
document_assembler,
sentenceDetectorDL,
tokenizer,
word_embeddings,
posology_ner,
ner_converter_icd,
c2doc,
sbert_embedder,
ndc_resolver
))

val clinical_note = Seq("""The patient was transferred secondary to inability and continue of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given aspirin 81 mg, folic acid 1 g daily, insulin glargine 100 UNT/ML injection and metformin 500 mg p.o. p.r.n.""").toDS.toDF("text")

val result = resolver_pipeline.fit(clinical_note).transform(clinical_note)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.ndc").predict("""The patient was transferred secondary to inability and continue of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given aspirin 81 mg, folic acid 1 g daily, insulin glargine 100 UNT/ML injection and metformin 500 mg p.o. p.r.n.""")
```

</div>


## Results


```bash
+-------------------------------------+------+-----------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                            ner_chunk|entity|   ndc_code|                                                                   description|                                                                                                                                                                                               all_codes|                                                                                                                                                                                         all_resolutions|                                                                                                                                                                                         other ndc codes|
+-------------------------------------+------+-----------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                        aspirin 81 mg|  DRUG|73089008114|                               aspirin 81 mg/81mg, 81 mg in 1 carton , capsule|[73089008114, 71872708704, 71872715401, 68210101500, 69536028110, 63548086706, 71679001000, 68196090051, 00113400500, 69536018112, 73089008112, 63981056362, 63739043402, 63548086705, 00113046708, 7...|[aspirin 81 mg/81mg, 81 mg in 1 carton , capsule, aspirin 81 mg 81 mg/1, 4 blister pack in 1 bag , tablet, aspirin 81 mg/1, 1 blister pack in 1 bag , tablet, coated, aspirin 81 mg/1, 1 bag in 1 dru...|         [-, -, -, -, -, -, -, -, -, -, -, 63940060962, -, -, -, -, -, -, -, -, 70000042002|00363021879|41250027408|36800046708|59779027408|49035027408|71476010131|81522046708|30142046708, -, -, -, -]|
|                       folic acid 1 g|  DRUG|43744015101|                                   folic acid 1 g/g, 1 g in 1 package , powder|[43744015101, 63238340000, 66326050555, 51552041802, 51552041805, 63238340001, 81919000204, 51552041804, 66326050556, 51552106301, 51927003300, 71092997701, 51927296300, 51552146602, 61281900002, 6...|[folic acid 1 g/g, 1 g in 1 package , powder, folic acid 1 kg/kg, 1 kg in 1 bottle , powder, folic acid 1 kg/kg, 1 kg in 1 drum , powder, folic acid 1 g/g, 5 g in 1 container , powder, folic acid 1...|                                                                                               [-, -, -, -, -, -, -, -, -, -, -, 51552139201, -, -, -, 81919000203, -, 81919000201, -, -, -, -, -, -, -]|
|insulin glargine 100 UNT/ML injection|  DRUG|00088502101|insulin glargine 100 [iu]/ml, 1 vial, glass in 1 package , injection, solution|[00088502101, 00088222033, 49502019580, 00002771563, 00169320111, 00088250033, 70518139000, 00169266211, 50090127600, 50090407400, 00002771559, 00002772899, 70518225200, 70518138800, 00024592410, 0...|[insulin glargine 100 [iu]/ml, 1 vial, glass in 1 package , injection, solution, insulin glargine 100 [iu]/ml, 1 vial, glass in 1 carton , injection, solution, insulin glargine 100 [iu]/ml, 1 vial ...|[-, -, -, 00088221900, -, -, 50090139800|00088502005, -, 70518146200|00169368712, 00169368512|73070020011, 00088221905|49502019675|50090406800, -, 73070010011|00169750111|50090495500, 66733077301|0...|
|                     metformin 500 mg|  DRUG|70010006315|               metformin hydrochloride 500 mg/500mg, 500 mg in 1 drum , tablet|[70010006315, 62207041613, 71052050750, 62207049147, 71052091050, 25000010197, 25000013498, 25000010198, 71052063005, 51662139201, 70010049118, 70882012456, 71052011005, 71052065905, 71052050850, 1...|[metformin hydrochloride 500 mg/500mg, 500 mg in 1 drum , tablet, metformin hcl 500 mg/kg, 50 kg in 1 drum , powder, 5-fluorouracil 500 g/500g, 500 g in 1 container , powder, metformin er 500 mg 50...|                                                                                             [-, -, -, 70010049105, -, -, -, -, -, -, -, -, -, -, -, 71800000801|42571036007, -, -, -, -, -, -, -, -, -]|
+-------------------------------------+------+-----------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_ndc|
|Compatibility:|Healthcare NLP 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[ndc_code]|
|Language:|en|
|Case sensitive:|false|



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIzNTUzODEwNl19
-->