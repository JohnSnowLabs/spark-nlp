---
layout: model
title: Sentence Entity Resolver for NCI-t
author: John Snow Labs
name: sbiobertresolve_ncit
date: 2023-03-26
tags: [entity_resolution, clinical, en, licensed, ncit]
task: Entity Resolution
language: en
edition: Healthcare NLP 4.3.2
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities related to clinical care, translational and basic research, public information and administrative activities to NCI-t codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

## Predicted Entities

`NCI-t codes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_ncit_en_4.3.2_3.0_1679843528109.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_ncit_en_4.3.2_3.0_1679843528109.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
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

ner_model = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
		.setInputCols(["sentence", "token", "embeddings"]) \
		.setOutputCol("ner")

ner_converter = NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\
    .setPreservePosition(False)

chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel\
  .pretrained("sbiobertresolve_ncit","en", "clinical/models") \
  .setInputCols(["ner_chunk", "sbert_embeddings"]) \
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")


nlpPipeline = Pipeline(stages=[document_assembler, 
                               sentence_detector, 
                               tokenizer, 
                               word_embeddings, 
                               ner_model, 
                               ner_converter, 
                               chunk2doc, 
                               sbert_embedder, 
                               resolver])

data = spark.createDataFrame([["""45 years old patient had Percutaneous mitral valve repair. He had Pericardiectomy 2 years ago. He has left cardiac ventricular systolic dysfunction in his history."""]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
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
  .setInputCols(Array("sentence","token"))
  .setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence","token","embeddings"))
  .setOutputCol("ner")

val ner_converter = new NerConverter()
  .setInputCols(Array("sentence","token","ner"))
  .setOutputCol("ner_chunk")

val chunk2doc = new Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
  .pretrained("sbiobert_base_cased_mli","en","clinical/models")
  .setInputCols(Array("ner_chunk_doc"))
  .setOutputCol("sbert_embeddings")

val resolver = SentenceEntityResolverModel
  .pretrained("sbiobertresolve_ncit","en", "clinical/models") 
  .setInputCols(Array("ner_chunk", "sbert_embeddings")) 
  .setOutputCol("resolution")
  .setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver))

val data = Seq("45 years old patient had Percutaneous mitral valve repair. He had Pericardiectomy 2 years ago. He has left cardiac ventricular systolic dysfunction in his history.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|   sent_id | ner_chunk                                     | entity    | NCI-t Code   | all_codes                                          | resolutions                                                                                                                              |
|----------:|:----------------------------------------------|:----------|:-------------|:---------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|
|         0 | Percutaneous mitral valve repair              | TREATMENT | C100003      | ['C100003', 'C158019', 'C80449', 'C50818', 'C80448'| ['percutaneous mitral valve repair [percutaneous mitral valve repair]', 'transcatheter mitral valve repair [transcatheter mitral valve...|
|         1 | Pericardiectomy                               | TREATMENT | C51643       | ['C51643', 'C51618', 'C100004', 'C62550', 'C51791' | ['pericardiectomy [pericardiectomy]', 'pericardiostomy [pericardiostomy]', 'pericardial stripping [pericardial stripping]', 'pulpectom...|
|         2 | left cardiac ventricular systolic dysfunction | PROBLEM   | C64251       | ['C64251', 'C146719', 'C55062', 'C50629', 'C111655'| ['left cardiac ventricular systolic dysfunction [left cardiac ventricular systolic dysfunction]', 'left ventricular systolic dysfuncti...|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_ncit|
|Compatibility:|Healthcare NLP 4.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[nci-t_code]|
|Language:|en|
|Size:|1.5 GB|
|Case sensitive:|false|

## References

https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/