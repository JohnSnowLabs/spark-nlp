---
layout: model
title: Zero-shot Relation Extraction (BioBert)
author: John Snow Labs
name: re_zeroshot_biobert
date: 2022-04-05
tags: [zero, shot, zero_shot, en, licensed]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.5.0
spark_version: 3.0
supported: true
annotator: ZeroShotRelationExtractionModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Zero-shot Relation Extraction to extract relations between clinical entities with no training dataset, just pretrained BioBert embeddings (included in the model). This model requires Healthcare NLP 3.5.0. 

Take a look at how it works in the "Open in Colab" section below.

## Predicted Entities




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.3.ZeroShot_Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_zeroshot_biobert_en_3.5.0_3.0_1649176740466.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
{% raw %}

```python
documenter = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = nlp.Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("tokens")

sentencer = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

words_embedder = nlp.WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

ner_clinical = medical.NerModel() \
    .pretrained("ner_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens", "embeddings"]) \
    .setOutputCol("ner_clinical")

ner_clinical_converter = nlp.NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_clinical"]) \
    .setOutputCol("ner_clinical_chunks")\
    .setWhiteList(["PROBLEM", "TEST"])      # PROBLEM-TEST-TREATMENT

ner_posology = medical.NerModel.pretrained("ner_posology", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens", "embeddings"]) \
    .setOutputCol("ner_posology")           

ner_posology_converter = nlp.NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_posology"]) \
    .setOutputCol("ner_posology_chunks")\
    .setWhiteList(["DRUG"])                # DRUG-FREQUENCY-DOSAGE-DURATION-FORM-ROUTE-STRENGTH

chunk_merger = medical.ChunkMergeApproach()\
    .setInputCols("ner_clinical_chunks", "ner_posology_chunks")\
    .setOutputCol('merged_ner_chunks')

## ZERO-SHOT RE Starting...

re_model = medical.ZeroShotRelationExtractionModel.pretrained("re_zeroshot_biobert", "en", "clinical/models")\
    .setInputCols(["merged_ner_chunks", "sentences"]) \
    .setOutputCol("relations")\
    .setMultiLabel(True)

re_model.setRelationalCategories({
    "ADE": ["{DRUG} causes {PROBLEM}."],
    "IMPROVE": ["{DRUG} improves {PROBLEM}.", "{DRUG} cures {PROBLEM}."],
    "REVEAL": ["{TEST} reveals {PROBLEM}."]})

pipeline = nlp.Pipeline() \
    .setStages([documenter, 
                tokenizer, 
                sentencer, 
                words_embedder, 
                ner_clinical, 
                ner_clinical_converter,
                ner_posology,
                ner_posology_converter,
                chunk_merger,
                re_model])

data = spark.createDataFrame( [["Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer."]] ).toDF("text")

results = pipeline.fit(data).transform(data)

results\
.selectExpr("explode(relations) as relation")\
.show(truncate=False)
```
```scala
val data = spark.createDataFrame(Seq("Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer.")).toDF("text")

val documenter = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("tokens")

val sentencer = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val ner_clinical = MedicalNerModel()
    .pretrained("ner_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_clinical")

val ner_clinical_converter = new NerConverter()
    .setInputCols(Array("sentences", "tokens", "ner_clinical"))
    .setOutputCol("ner_clinical_chunks")
    .setWhiteList(Array("PROBLEM", "TEST"))      # PROBLEM-TEST-TREATMENT

val ner_posology = MedicalNerModel()
    .pretrained("ner_posology", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_posology")           

val ner_posology_converter = new NerConverter()
    .setInputCols(Array("sentences", "tokens", "ner_posology"))
    .setOutputCol("ner_posology_chunks")
    .setWhiteList(Array("DRUG"))                # DRUG-FREQUENCY-DOSAGE-DURATION-FORM-ROUTE-STRENGTH

val chunk_merger = ChunkMergeApproach()
    .setInputCols("ner_clinical_chunks", "ner_posology_chunks")
    .setOutputCol('merged_ner_chunks')

## ZERO-SHOT RE Starting...

val re_model = ZeroShotRelationExtractionModel.pretrained("re_zeroshot_biobert", "en", "clinical/models")
    .setInputCols(Array("ner_chunks", "sentences"))
    .setOutputCol("relations")
    .setMultiLabel(true)

re_model.setRelationalCategories({ Map(
    "CURE" -> Array("{TREATMENT} cures {PROBLEM}."),
    "IMPROVE" -> Array("{TREATMENT} improves {PROBLEM}.", "{TREATMENT} cures {PROBLEM}."),
    "REVEAL" -> Array("{TEST} reveals {PROBLEM}.") ))

val pipeline = new Pipeline()
    .setStages(Array(documenter, 
                     tokenizer, 
                     sentencer, 
                     words_embedder, 
                     ner_clinical, 
                     ner_clinical_converter,
                     ner_posology,
                     ner_posology_converter,
                     chunk_merger,
                     re_model))

val model = pipeline.fit(data)
val results = model.transform(data)
```
{% endraw %}


{:.nlu-block}
```python
import nlu
nlu.load("en.relation.zeroshot_biobert").predict("""Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer.""")
```

</div>


## Results


```bash
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|relation                                                                                                                                                                                                                                                                                                                                                                 |
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|{category, 268, 358, IMPROVE, {entity1_begin -> 0, relation -> IMPROVE, hypothesis -> Paracetamol improves sickness., confidence -> 0.98819494, nli_prediction -> entail, entity1 -> DRUG, syntactic_distance -> undefined, chunk2 -> sickness, entity2_end -> 45, entity1_end -> 10, entity2_begin -> 38, entity2 -> PROBLEM, chunk1 -> Paracetamol, sentence -> 0}, []}|
|{category, 0, 90, IMPROVE, {entity1_begin -> 0, relation -> IMPROVE, hypothesis -> Paracetamol improves headache., confidence -> 0.9929625, nli_prediction -> entail, entity1 -> DRUG, syntactic_distance -> undefined, chunk2 -> headache, entity2_end -> 33, entity1_end -> 10, entity2_begin -> 26, entity2 -> PROBLEM, chunk1 -> Paracetamol, sentence -> 0}, []}    |
|{category, 536, 615, REVEAL, {entity1_begin -> 48, relation -> REVEAL, hypothesis -> An MRI test reveals cancer., confidence -> 0.9760039, nli_prediction -> entail, entity1 -> TEST, syntactic_distance -> undefined, chunk2 -> cancer, entity2_end -> 85, entity1_end -> 58, entity2_begin -> 80, entity2 -> PROBLEM, chunk1 -> An MRI test, sentence -> 1}, []}       |
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|re_zeroshot_biobert|
|Compatibility:|Healthcare NLP 3.5.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|


## References


As it is a zero-shot relation extractor, no training dataset is necessary.
