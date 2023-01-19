---
layout: model
title: Normalizing Section Headers in Clinical Notes
author: John Snow Labs
name: normalized_section_header_mapper
date: 2022-04-01
tags: [en, chunkmapper, chunkmapping, normalizer, sectionheader, licensed, clinical]
task: Chunk Mapping
language: en
edition: Healthcare NLP 3.5.0
spark_version: 3.0
supported: true
annotator: NotDefined
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This pretrained pipeline normalizes the section headers in clinical notes. It returns two levels of normalization called `level_1` and `level_2`.


## Predicted Entities

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NORMALIZED_SECTION_HEADER_MAPPER/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NORMALIZED_SECTION_HEADER_MAPPER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/normalized_section_header_mapper_en_3.5.04.2_3.0_16488194861909098646707.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/normalized_section_header_mapper_en_3.5.04.2_3.0_16488194861909098646707.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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


sentences = """ADMISSION DIAGNOSIS Right pleural effusion and suspected malignant mesothelioma.
PRINCIPAL DIAGNOSIS Right pleural effusion, suspected malignant mesothelioma.
GENERAL REVIEW Right pleural effusion, firm nodules, diffuse scattered throughout the right pleura and diaphragmatic surface."""

test_data = spark.createDataFrame([[sentences]]).toDF("text")

result = pipeline.fit(test_data).transform(test_data)
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


val test_data = Seq(test_sentence).toDS.toDF("text")
val result = pipeline.fit(test_data).transform(test_data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.section_headers_normalized").predict("""ADMISSION DIAGNOSIS Right pleural effusion and suspected malignant mesothelioma.
PRINCIPAL DIAGNOSIS Right pleural effusion, suspected malignant mesothelioma.
GENERAL REVIEW Right pleural effusion, firm nodules, diffuse scattered throughout the right pleura and diaphragmatic surface.""")
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
|Compatibility:|Healthcare NLP 3.5.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|14.2 KB|
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTQ4NTk2MTA1Ml19
-->