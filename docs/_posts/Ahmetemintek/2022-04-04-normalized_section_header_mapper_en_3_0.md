---
layout: model
title: Normalizing Section Headers in Clinical Notes
author: John Snow Labs
name: normalized_section_header_mapper
date: 2022-04-04
tags: [en, chunkmapper, chunkmapping, normalizer, sectionheader, licensed]
task: Chunk Mapping
language: en
edition: Spark NLP for Healthcare 3.4.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline normalizes the section headers in clinical notes. It returns two levels of normalization called `level_1` and `level_2`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/normalized_section_header_mapper_en_3.4.2_3.0_1649098646707.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

The sample code snippet may not contain all required fields of a pipeline. In this case, you can reach out a related colab notebook containing the end-to-end pipeline and more by clicking the "Open in Colab" link above.




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

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en","clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_jsl_slim", "en", "clinical/models")\
      .setInputCols(["sentence","token", "word_embeddings"])\
      .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")\
      .setWhiteList(["Header"])

chunkerMapper = ChunkMapperModel.pretrained("normalized_section_header_mapper", "en", "clinical/models") \
      .setInputCols("ner_chunk")\
      .setOutputCol("mappings")\
      .setRel("level_1") #or level_2

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 embeddings,
                                 clinical_ner, 
                                 ner_converter, 
                                 chunkerMapper])

sentences = [
    ["""ADMISSION DIAGNOSIS Right pleural effusion and suspected malignant mesothelioma.
        PRINCIPAL DIAGNOSIS Right pleural effusion, suspected malignant mesothelioma.
        GENERAL REVIEW Right pleural effusion, firm nodules, diffuse scattered throughout the right pleura and diaphragmatic surface.
     """]]

test_data = spark.createDataFrame([sentences]).toDF("text")
res = pipeline.fit(test_data).transform(test_data)
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

val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en","clinical/models")
        .setInputCols(Array("sentence", "token"))
        .setOutputCol("word_embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_jsl_slim", "en", "clinical/models")
        .setInputCols(Array("sentence","token", "word_embeddings"))
        .setOutputCol("ner")

val ner_converter = new NerConverter()
        .setInputCols(Array("sentence", "token", "ner"))
        .setOutputCol("ner_chunk")
        .setWhiteList(Array("Header"))

val chunkerMapper = ChunkMapperModel.pretrained("normalized_section_header_mapper", "en", "clinical/models") 
        .setInputCols("ner_chunk")
        .setOutputCol("mappings")
        .setRel("level_1") #or level_2

val pipeline = new Pipeline().setStages(Array(document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 embeddings,
                                 clinical_ner, 
                                 ner_converter, 
                                 chunkerMapper))

val test_sentence= """ADMISSION DIAGNOSIS Right pleural effusion and suspected malignant mesothelioma.
                                      PRINCIPAL DIAGNOSIS Right pleural effusion, suspected malignant mesothelioma.
                                     GENERAL REVIEW Right pleural effusion, firm nodules, diffuse scattered throughout the right pleura and diaphragmatic surface."""

val test_data = Seq(test_sentence).toDF("text")
val res = pipeline.fit(test_data).transform(test_data)
```
</div>

## Results

```bash
+-------------------+------------------+
|section            |normalized_section|
+-------------------+------------------+
|ADMISSION DIAGNOSIS|DIAGNOSIS         |
|PRINCIPAL DIAGNOSIS|DIAGNOSIS         |
|GENERAL REVIEW     |REVIEW TYPE       |
+-------------------+------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|normalized_section_header_mapper|
|Compatibility:|Spark NLP for Healthcare 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|14.2 KB|
