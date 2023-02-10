---
layout: model
title: Mapping Drugs With Their Corresponding Adverse Drug Events (ADE)
author: John Snow Labs
name: drug_ade_mapper
date: 2022-08-23
tags: [en, chunkmapping, chunkmapper, drug, ade, licensed]
task: Chunk Mapping
language: en
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
recommended: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps drugs with their corresponding Adverse Drug Events.

## Predicted Entities

`ADE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/drug_ade_mapper_en_4.0.2_3.0_1661250246683.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/drug_ade_mapper_en_4.0.2_3.0_1661250246683.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

sentence_detector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

#NER model to detect drug in the text
ner = MedicalNerModel.pretrained('ner_posology_greedy', 'en', 'clinical/models') \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")

ner_chunk = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")\

chunkMapper = ChunkMapperModel.pretrained("drug_ade_mapper", "en", "clinical/models")\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setRels(["ADE"])

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 ner, 
                                 ner_chunk, 
                                 chunkMapper])

text = ["""The patient was prescribed 1000 mg fish oil and multivitamins. 
            She was discharged on zopiclone and ambrisentan"""]

data = spark.createDataFrame([text]).toDF("text")

result= pipeline.fit(data).transform(data)

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

#NER model to detect drug in the text
val ner = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models") 
      .setInputCols(Array("sentence", "token", "embeddings"))
      .setOutputCol("ner")

val ner_chunk = new NerConverter() 
      .setInputCols(Array("sentence", "token", "ner")) 
      .setOutputCol("ner_chunk")

val chunkMapper = ChunkMapperModel.pretrained("drug_ade_mapper", "en", "clinical/models")
      .setInputCols(Array("ner_chunk"))
      .setOutputCol("mappings")
      .setRels(Array("ADE"))

val pipeline = new Pipeline(stages = Array(
                                 document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 ner, 
                                 ner_chunk, 
                                 chunkMapper))

val data = Seq("The patient was prescribed 1000 mg fish oil and multivitamins. She was discharged on zopiclone and ambrisentan").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------+------------+-------------------------------------------------------------------------------------------+
|ner_chunk       |ade_mappings|all_relations                                                                              |
+----------------+------------+-------------------------------------------------------------------------------------------+
|1000 mg fish oil|Dizziness   |Myocardial infarction:::Nausea                                                             |
|multivitamins   |Erythema    |Acne:::Dry skin:::Skin burning sensation:::Inappropriate schedule of product administration|
|zopiclone       |Vomiting    |Malaise:::Drug interaction:::Asthenia:::Hyponatraemia                                      |
|ambrisentan     |Dyspnoea    |Therapy interrupted:::Death:::Dizziness:::Drug ineffective                                 |
+----------------+------------+-------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|drug_ade_mapper|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_pos_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|7.9 MB|

## References

Data from the FDA Adverse Event Reporting System (FAERS) for the years 2020, 2021 and 2022 were used as the source for this mapper model.

https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
