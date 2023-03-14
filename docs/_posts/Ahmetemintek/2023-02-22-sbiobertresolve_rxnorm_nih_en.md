---
layout: model
title: Sentence Entity Resolver for RxNorm According to National Institute of Health (NIH) Database (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_rxnorm_nih
date: 2023-02-22
tags: [entity_resolution, rxnorm, clinical, en, licensed]
task: Entity Resolution
language: en
nav_key: models
edition: Healthcare NLP 4.3.0
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes according to the National Institute of Health (NIH) database using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_nih_en_4.3.0_3.0_1677106956679.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_nih_en_4.3.0_3.0_1677106956679.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models") \
		.setInputCols(["sentence", "token", "embeddings"]) \
		.setOutputCol("ner")

ner_converter = NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\
    .setWhiteList(['DRUG'])\
    .setPreservePosition(False)

chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_nih","en", "clinical/models") \
    .setInputCols(["sbert_embeddings"]) \
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
                               rxnorm_resolver])

data = spark.createDataFrame([["""She is given folic acid 1 mg daily , levothyroxine 0.1 mg and aspirin 81 mg daily ."""]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
	.setInputCols("document")
	.setOutputCol("sentence")

val tokenizer = new Tokenizer()
	.setInputCols("sentence")
	.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("entities")

val chunk2doc = new Chunk2Doc()
    .setInputCols("ner_chunk")
    .setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")
    .setInputCols("ner_chunk_doc")
    .setOutputCol("sbert_embeddings")

val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_nih","en", "clinical/models")
    .setInputCols(Array("ner_chunk", "sbert_embeddings"))
    .setOutputCol("resolution")
    .setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner_model, ner_converter, chunk2doc, sbert_embedder, rxnorm_resolver))

val data = Seq("""She is given folic acid 1 mg daily , levothyroxine 0.1 mg and aspirin 81 mg daily and metformin 100 mg, coumadin 5 mg.""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    |   sent_id | ner_chunk            | entity   |   rxnorm_code | all_codes                                                   | resolutions                                                                                                                          |
|---:|----------:|:---------------------|:---------|--------------:|:------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|
|  0 |         0 | folic acid 1 mg      | DRUG     |      12281181 | ['12281181', '12283696', '12270292', '12306595',  1227889...| ['folic acid 1 MG [folic acid 1 MG]', 'folic acid 1.1 MG [folic acid 1.1 MG]', 'folic acid 1 MG/ML [folic acid 1 MG/ML]', 'folic a...|
|  1 |         0 | levothyroxine 0.1 mg | DRUG     |      12275630 | ['12275630', '12275646', '12301585', '12306484',  1235044...| ['levothyroxine sodium 0.1 MG [levothyroxine sodium 0.1 MG]', 'levothyroxine sodium 0.01 MG [levothyroxine sodium 0.01 MG]', 'levo...|
|  2 |         0 | aspirin 81 mg        | DRUG     |      12278696 | ['12278696', '12299811', '12298729', '12311168', '1230631...| ['aspirin 81 MG [aspirin 81 MG]', 'aspirin 81 MG [YSP Aspirin] [aspirin 81 MG [YSP Aspirin]]', 'aspirin 81 MG [Med Aspirin] [aspir...|
|  3 |         0 | metformin 100 mg     | DRUG     |      12282749 | ['12282749', '3735316', '12279966', '1509573', '3736179'... | ['metformin hydrochloride 100 MG/ML [metformin hydrochloride 100 MG/ML]', 'metFORMIN hydrochloride 100 MG/ML [metFORMIN hydrochlor...|
|  4 |         0 | coumadin 5 mg        | DRUG     |       1768579 | ['1768579', '12534260', '1780903', '1768951', '1510873' ... | ['coumarin 5 MG [coumarin 5 MG]', 'vericiguat 5 MG [vericiguat 5 MG]', 'pridinol 5 MG [pridinol 5 MG]', 'propinox 5 MG [propinox 5...|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_rxnorm_nih|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Size:|818.8 MB|
|Case sensitive:|false|

## References

Trained on February 2023 with `sbiobert_base_cased_mli` embeddings.
https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html