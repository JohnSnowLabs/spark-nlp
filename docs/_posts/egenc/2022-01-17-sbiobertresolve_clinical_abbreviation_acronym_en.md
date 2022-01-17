---
layout: model
title: Sentence Entity Resolver for Clinical Abbreviations and Acronyms (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_clinical_abbreviation_acronym
date: 2022-01-17
tags: [abbreviation, entity_resolver, licensed, clinical, acronym, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.3.4
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical abbreviations and acronyms to their meanings using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It is the first primitive version of abbreviation resolution and will be improved further in the following releases.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_clinical_abbreviation_acronym_en_3.3.4_2.4_1642426791378.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
      .setInputCols("document")\
      .setOutputCol("sentence")

tokenizer = Tokenizer() \
      .setInputCols(["document"]) \
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical','en', 'clinical/models')\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

ner = NerDLModel.pretrained("ner_radiology", "en", "clinical/models") \
       .setInputCols(["sentence", "token", "embeddings"]) \
       .setOutputCol("ner")

ner_converter = NerConverter() \
       .setInputCols(["sentence", "token", "ner"]) \
       .setOutputCol("ner_chunk")\
       .setWhiteList(['Test'])

c2doc = Chunk2Doc()\
       .setInputCols("merged_chunk")\
       .setOutputCol("ner_chunk_doc") 

sentence_chunk_embeddings = BertSentenceChunkEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
       .setInputCols(["document", "merged_chunk"])\
       .setOutputCol("sentence_embeddings")\
       .setChunkWeight(0.5)

abbr_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_clinical_abbreviation_acronym", "en", "clinical/models") \
       .setInputCols(["merged_chunk", "sentence_embeddings"]) \
       .setOutputCol("abbr_meaning")\
       .setDistanceFunction("EUCLIDEAN")\
       .setCaseSensitive(False)
    
resolver_pipeline = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        ner,
        ner_converter,
        c2doc,
        sentence_chunk_embeddings,
        abbr_resolver
  ])

model = resolver_pipeline.fit(spark.createDataFrame([['']]).toDF("text"))

sample_text = "HISTORY OF PRESENT ILLNESS: The patient three weeks ago was seen at another clinic for upper respiratory infection-type symptoms. She was diagnosed with a viral infection and had used OTC medications including Tylenol, Sudafed, and Nyquil."
abbr_result = model.transform(spark.createDataFrame([[text]]).toDF('text'))
```
```scala


val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")

val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("tokens")

val wordEmbeddings = BertEmbeddings
      .pretrained("biobert_pubmed_base_cased")
      .setInputCols(Array("sentence", "tokens"))
      .setOutputCol("word_embeddings")

val nerModel = NerDLModel
      .pretrained("ner_radiology", "en", "clinical/models")
      .setInputCols(Array("sentence", "tokens", "word_embeddings"))
      .setOutputCol("ner")

val nerConverter = new NerConverter()
      .setInputCols("sentence", "tokens", "ner")
      .setOutputCol("ner_chunk")


val c2doc = Chunk2Doc()
      .setInputCols("merged_chunk")
      .setOutputCol("ner_chunk_doc") 

val sentence_chunk_embeddings = BertSentenceChunkEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
      .setInputCols(Array("document", "merged_chunk"))
      .setOutputCol("sentence_embeddings")
      .setChunkWeight(0.5)

val abbr_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_clinical_abbreviation_acronym", "en", "clinical/models") 
      .setInputCols(Array("merged_chunk", "sentence_embeddings")) 
      .setOutputCol("abbr_meaning")
      .setDistanceFunction("EUCLIDEAN")
      .setCaseSensitive(False)
    
val resolver_pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, word_embeddings, clinical_ner, ner_converter_icd, entity_extractor, chunk_merge, c2doc, sentence_chunk_embeddings, abbr_resolver))

val sample_text = Seq("HISTORY OF PRESENT ILLNESS: The patient three weeks ago was seen at another clinic for upper respiratory infection-type symptoms. She was diagnosed with a viral infection and had used OTC medications including Tylenol, Sudafed, and Nyquil.").toDF("text")
val abbr_result = resolver_pipeline.fit(sample_text).transform(sample_text)
```
</div>

## Results

```bash
|   sent_id | ner_chunk   | entity   | abbr_meaning     | all_k_results                                                                      | all_k_resolutions          |
|----------:|:------------|:---------|:-----------------|:-----------------------------------------------------------------------------------|:---------------------------|
|         0 | OTC         | ABBR     | over the counter | ['over the counter', 'ornithine transcarbamoylase', 'enteric-coated', 'thyroxine'] | ['OTC', 'OTC', 'EC', 'T4'] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_clinical_abbreviation_acronym|
|Compatibility:|Spark NLP for Healthcare 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[abbr_meaning]|
|Language:|en|
|Size:|106.2 MB|
|Case sensitive:|false|