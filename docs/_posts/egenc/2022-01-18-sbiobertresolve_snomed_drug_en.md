---
layout: model
title: Sentence Entity Resolver for SNOMED (sbiobertresolve_snomed_drug)
author: John Snow Labs
name: sbiobertresolve_snomed_drug
date: 2022-01-18
tags: [licensed, snomed, clinical, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps detected drug entities to SNOMED codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

## Predicted Entities

`SNOMED Codes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_snomed_drug_en_3.3.4_2.4_1642534694043.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", 'clinical/models') \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
.setInputCols(["sentence", "token", "word_embeddings"]) \
.setOutputCol("ner")

ner_converter = NerConverterInternal() \
.setInputCols(["sentence", "token", "ner"]) \
.setOutputCol("ner_chunk")\
.setWhiteList(['DRUG'])

c2doc = Chunk2Doc()\
.setInputCols("ner_chunk")\
.setOutputCol("ner_chunk_doc") 

sentence_chunk_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sentence_embeddings")


snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_drug", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("snomed_code")\
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
sentence_chunk_embeddings,
snomed_resolver
])

model = resolver_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = model.transform(spark.createDataFrame([["She is given Fragmin 5000 units subcutaneously daily, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Avandia 4 mg daily, aspirin 81 mg daily, Neurontin 400 mg p.o. t.i.d., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin."]]).toDF("text"))
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") 
.setInputCols(Array("document")) 
.setOutputCol("sentence")

val tokenizer = Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("word_embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") 
.setInputCols(Array("sentence", "token", "word_embeddings")) 
.setOutputCol("ner")

val ner_converter = NerConverterInternal() 
.setInputCols(Array("sentence", "token", "ner")) 
.setOutputCol("ner_chunk")\
.setWhiteList(Array("DRUG"))

val c2doc = Chunk2Doc()
.setInputCols("ner_chunk")
.setOutputCol("ner_chunk_doc") 

val sentence_chunk_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sentence_embeddings")


val snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_drug", "en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sentence_embeddings")) 
.setOutputCol("snomed_code")
.setDistanceFunction("EUCLIDEAN")


resolver_pipeline = new Pipeline().setStages(document_assembler, sentenceDetectorDL, tokenizer, word_embeddings, clinical_ner, ner_converter, c2doc, sentence_chunk_embeddings, snomed_resolver)
data = Seq("She is given Fragmin 5000 units subcutaneously daily, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Avandia 4 mg daily, aspirin 81 mg daily, Neurontin 400 mg p.o. t.i.d., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin.").toDF("text")

result = resolver_pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.snomed_drug").predict("""She is given Fragmin 5000 units subcutaneously daily, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Avandia 4 mg daily, aspirin 81 mg daily, Neurontin 400 mg p.o. t.i.d., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin.""")
```

</div>

## Results

```bash
+-----------------+------+-----------------+-----------------+------------------------------------------------------------+------------------------------------------------------------+
|        ner_chunk|entity|      snomed_code|    resolved_text|                                               all_k_results|                                           all_k_resolutions|
+-----------------+------+-----------------+-----------------+------------------------------------------------------------+------------------------------------------------------------+
|          Fragmin|  DRUG| 9487801000001106|          Fragmin|9487801000001106:::130752006:::28999000:::953500100000110...|Fragmin:::Fragilysin:::Fusarin:::Femulen:::Fumonisin:::Fr...|
|        OxyContin|  DRUG| 9296001000001100|        OxyCONTIN|9296001000001100:::373470001:::230091000001108:::55452001...|OxyCONTIN:::Oxychlorosene:::Oxyargin:::oxyCODONE:::Oxymor...|
|       folic acid|  DRUG|         63718003|       Folic acid|63718003:::6247001:::226316008:::432165000:::438451000124...|Folic acid:::Folic acid-containing product:::Folic acid s...|
|    levothyroxine|  DRUG|10071011000001106|    Levothyroxine|10071011000001106:::710809001:::768532006:::126202002:::7...|Levothyroxine:::Levothyroxine (substance):::Levothyroxine...|
|          Avandia|  DRUG| 9217601000001109|          avandia|9217601000001109:::9217501000001105:::12226401000001108::...|avandia:::avandamet:::Anatera:::Intanza:::Avamys:::Aragam...|
|          aspirin|  DRUG|        387458008|          Aspirin|387458008:::7947003:::5145711000001107:::426365001:::4125...|Aspirin:::Aspirin-containing product:::Aspirin powder:::A...|
|        Neurontin|  DRUG| 9461401000001102|        neurontin|9461401000001102:::130694004:::86822004:::952840100000110...|neurontin:::Neurolysin:::Neurine (substance):::Nebilet:::...|
|magnesium citrate|  DRUG|         12495006|Magnesium citrate|12495006:::387401007:::21691008:::15531411000001106:::408...|Magnesium citrate:::Magnesium carbonate:::Magnesium trisi...|
|          insulin|  DRUG|         67866001|          Insulin|67866001:::325072002:::414515005:::39487003:::411530000::...|Insulin:::Insulin aspart:::Insulin detemir:::Insulin-cont...|
+-----------------+------+-----------------+-----------------+------------------------------------------------------------+------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_snomed_drug|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[snomed_code]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|false|
|Dependencies:|ner_posology|

## Data Source

Trained on `SNOMED` code dataset with `sbiobert_base_cased_mli` sentence embeddings.