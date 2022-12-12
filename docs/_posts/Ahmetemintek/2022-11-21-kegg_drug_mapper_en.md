---
layout: model
title: Mapping Drugs from the KEGG Database to Their Efficacies, Molecular Weights and Corresponding Codes from Other Databases
author: John Snow Labs
name: kegg_drug_mapper
date: 2022-11-21
tags: [drug, efficacy, molecular_weight, cas, pubchem, chebi, ligandbox, nikkaji, pdbcct, chunk_mapper, clinical, en, licensed]
task: Chunk Mapping
language: en
edition: Healthcare NLP 4.2.2
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps drugs with their corresponding `efficacy`, `molecular_weight` as well as `CAS`, `PubChem`, `ChEBI`, `LigandBox`, `NIKKAJI`, `PDB-CCD` codes. This model was trained with the data from the KEGG database.

## Predicted Entities

`efficacy`, `molecular_weight`, `CAS`, `PubChem`, `ChEBI`, `LigandBox`, `NIKKAJI`, `PDB-CCD`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/kegg_drug_mapper_en_4.2.2_3.0_1669069910375.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\

chunkerMapper = ChunkMapperModel.pretrained("kegg_drug_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["efficacy", "molecular_weight", "CAS", "PubChem", "ChEBI", "LigandBox", "NIKKAJI", "PDB-CCD"])\

pipeline = Pipeline().setStages([
    document_assembler,
    sentence_detector,
    tokenizer, 
    word_embeddings,
    ner, 
    converter, 
    chunkerMapper])


text= "She is given OxyContin, folic acid, levothyroxine, Norvasc, aspirin, Neurontin"

data = spark.createDataFrame([[text]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("ner")

val converter = new NerConverter() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")

val chunkerMapper = ChunkMapperModel.pretrained("kegg_drug_mapper", "en", "clinical/models")
    .setInputCols("ner_chunk")
    .setOutputCol("mappings")
    .setRels(Array(["efficacy", "molecular_weight", "CAS", "PubChem", "ChEBI", "LigandBox", "NIKKAJI", "PDB-CCD"]))


val pipeline = new Pipeline().setStages(Array(
    document_assembler,
    sentence_detector,
    tokenizer, 
    word_embeddings,
    ner, 
    converter, 
    chunkerMapper))


val text= "She is given OxyContin, folic acid, levothyroxine, Norvasc, aspirin, Neurontin"

val data = Seq(text).toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------+--------------------------------------------------+----------------+----------+-----------+-------+---------+---------+-------+
|    ner_chunk|                                          efficacy|molecular_weight|       CAS|    PubChem|  ChEBI|LigandBox|  NIKKAJI|PDB-CCD|
+-------------+--------------------------------------------------+----------------+----------+-----------+-------+---------+---------+-------+
|    OxyContin|     Analgesic (narcotic), Opioid receptor agonist|        351.8246|  124-90-3|  7847912.0| 7859.0|   D00847|J281.239H|   NONE|
|   folic acid|Anti-anemic, Hematopoietic, Supplement (folic a...|        441.3975|   59-30-3|  7847138.0|27470.0|   D00070|  J1.392G|    FOL|
|levothyroxine|                     Replenisher (thyroid hormone)|          776.87|   51-48-9|9.6024815E7|18332.0|   D08125|  J4.118A|    T44|
|      Norvasc|Antihypertensive, Vasodilator, Calcium channel ...|        408.8759|88150-42-9|5.1091781E7| 2668.0|   D07450| J33.383B|   NONE|
|      aspirin|Analgesic, Anti-inflammatory, Antipyretic, Anti...|        180.1574|   50-78-2|  7847177.0|15365.0|   D00109|  J2.300K|    AIN|
|    Neurontin|                     Anticonvulsant, Antiepileptic|        171.2368|60142-96-3|  7847398.0|42797.0|   D00332| J39.388F|    GBN|
+-------------+--------------------------------------------------+----------------+----------+-----------+-------+---------+---------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kegg_drug_mapper|
|Compatibility:|Healthcare NLP 4.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|1.0 MB|