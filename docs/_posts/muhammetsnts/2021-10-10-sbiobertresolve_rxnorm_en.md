---
layout: model
title: Sentence Entity Resolver for RxNorm (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_rxnorm
date: 2021-10-10
tags: [rxnorm, entity_resolution, licensed, clinical, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.2.3
spark_version: 2.4
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using `sbiobert_base_cased_mli ` Sentence Bert Embeddings.

## Predicted Entities

`RxNorm Codes`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_RXNORM/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_en_3.2.3_2.4_1633875017884.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
document_assembler = DocumentAssembler()\
		.setInputCol("text")\
		.setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") \
		.setInputCols(["document"]) \
		.setOutputCol("sentence")

tokenizer = Tokenizer()\
		.setInputCols(["sentence"])\
		.setOutputCol("token")
	
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
		.setInputCols(["sentence", "token"])\
		.setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
		.setInputCols(["sentence", "token", "embeddings"]) \
		.setOutputCol("jsl_ner")

ner_converter = NerConverter() \
		.setInputCols(["sentence", "token", "jsl_ner"]) \
		.setOutputCol("ner_chunk")


chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

rxnorm_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_rxnorm","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, rxnorm_resolver])

data = spark.createDataFrame([["""She is given Fragmin 5000 units subcutaneously daily , Xenaderm to wounds topically b.i.d., lantus 40 units subcutaneously at bedtime , OxyContin 30 mg p.o.q. , folic acid 1 mg daily , levothyroxine 0.1 mg 
p.o. daily , Prevacid 30 mg daily , Avandia 4 mg daily , norvasc 10 mg daily , lexapro 20 mg daily , aspirin 81 mg daily , Neurontin 400 mg ."""]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
...
val document_assembler = new DocumentAssembler()
		.setInputCol("text")
		.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
		.setInputCols(Array("document"))
		.setOutputCol("sentence")

val tokenizer = new Tokenizer()
		.setInputCols(Array("sentence"))
		.setOutputCol("token")
	
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
		.setInputCols(Array("sentence", "token"))
	    	.setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models")
		.setInputCols(Array("sentence", "token", "embeddings"))
		.setOutputCol("jsl_ner")

val ner_converter = new NerConverter()
		.setInputCols(Array("sentence", "token", "jsl_ner"))
		.setOutputCol("ner_chunk")

val chunk2doc = new Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val rxnorm_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_rxnorm","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, rxnorm_resolver))

val data = Seq("She is given Fragmin 5000 units subcutaneously daily , Xenaderm to wounds topically b.i.d., lantus 40 units subcutaneously at bedtime , OxyContin 30 mg p.o.q. , folic acid 1 mg daily , levothyroxine 0.1 mg p.o. daily , Prevacid 30 mg daily , Avandia 4 mg daily , norvasc 10 mg daily , lexapro 20 mg daily , aspirin 81 mg daily , Neurontin 400 mg .").toDF("text")

val result = pipeline.fit(data).transform(data)

```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.rxnorm").predict("""She is given Fragmin 5000 units subcutaneously daily , Xenaderm to wounds topically b.i.d., lantus 40 units subcutaneously at bedtime , OxyContin 30 mg p.o.q. , folic acid 1 mg daily , levothyroxine 0.1 mg 
p.o. daily , Prevacid 30 mg daily , Avandia 4 mg daily , norvasc 10 mg daily , lexapro 20 mg daily , aspirin 81 mg daily , Neurontin 400 mg .""")
```

</div>

## Results

```bash
+-------------+------+----------+----------------------------------------------------------------------+----------------------------------------------------------------------+
|    ner_chunk|entity|icd10_code|                                                             all_codes|                                                           resolutions|
+-------------+------+----------+----------------------------------------------------------------------+----------------------------------------------------------------------+
|      Fragmin|  DRUG|    281554|281554:::1992532:::151794:::217814:::361779:::1098701:::203870:::15...|Fragmin:::Kedrab:::Frisium:::Isopto Frin:::Frumax:::Folgard:::Faslo...|
|     Xenaderm|  DRUG|    581754|581754:::898496:::202363:::1307304:::94611:::1046399:::1093360:::11...|Xenaderm:::Xiaflex:::Xanax:::Xtandi:::Xerac AC:::Xgeva:::Xoten:::Xa...|
|       lantus|  DRUG|    261551|261551:::151959:::28381:::202990:::196502:::608814:::1040032:::7049...|Lantus:::Laratrim:::lachesine:::Larodopa:::Lamictal:::Lansinoh:::La...|
|    OxyContin|  DRUG|    218986|218986:::32680:::32698:::352843:::218859:::1086614:::1120014:::2189...|Oxycontin:::oxychlorosene:::oxyphencyclimine:::Ocutricin HC:::Ocutr...|
|   folic acid|  DRUG|      4511|4511:::1162058:::1162059:::62356:::1376005:::542060:::619039:::1162...|folic acid:::folic acid Oral Product:::folic acid Pill:::folate:::F...|
|levothyroxine|  DRUG|     10582|10582:::1868004:::40144:::1602753:::1602745:::227577:::1602750:::11...|levothyroxine:::levothyroxine Injection:::levothyroxine sodium:::le...|
|     Prevacid|  DRUG|     83156|83156:::219485:::219171:::606051:::858359:::1547099:::2286610:::105...|Prevacid:::Provisc:::Perisine:::ProQuad:::Acuvail:::suvorexant:::Pi...|
|      Avandia|  DRUG|    261455|261455:::152800:::1310526:::236219:::1370666:::686438:::215221:::99...|Avandia:::Amilamont:::Aubagio:::alibendol:::anisate:::Invega:::Amil...|
|      norvasc|  DRUG|     58927|58927:::1876388:::218772:::262324:::385700:::226108:::203013:::2036...|Norvasc:::NoRisc:::Norco:::Norflex:::Norval:::Norimode:::Norcuron::...|
|      lexapro|  DRUG|    352741|352741:::580253:::227285:::2058916:::24867:::847463:::2055761:::144...|Lexapro:::Levsinex:::Loprox:::Vizimpro:::fenproporex:::Levoprome:::...|
|      aspirin|  DRUG|      1191|1191:::405403:::218266:::1154070:::215568:::202547:::1154069:::2393...|aspirin:::YSP Aspirin:::Med Aspirin:::aspirin Pill:::Bayer Aspirin:...|
|    Neurontin|  DRUG|    196498|196498:::152627:::151178:::827343:::134802:::1720602:::152141:::131...|Neurontin:::Nystamont:::Nitronal:::Nucort:::Naropin:::Nucala:::Nyst...|
+-------------+------+----------+----------------------------------------------------------------------+----------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_rxnorm|
|Compatibility:|Healthcare NLP 3.2.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on 02 August 2021 RxNorm dataset.
