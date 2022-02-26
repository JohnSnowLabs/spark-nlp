---
layout: model
title: Sentence Entity Resolver for ATC (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_atc
date: 2022-02-26
tags: [atc, licensed, en, clinical, entity_resolution, open_source]
task: Entity Resolution
language: en
edition: Spark NLP 3.4.1
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps drugs entities to ATC (Anatomic Therapeutic Chemical) codes using sbiobert_base_cased_mli Sentence Bert Embeddings.

## Predicted Entities

`ATC Codes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbiobertresolve_atc_en_3.4.1_2.4_1645879811268.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

posology_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "word_embeddings"]) \
      .setOutputCol("ner")

ner_converter = NerConverterInternal() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")\
      .setWhiteList(['DRUG'])

c2doc = Chunk2Doc()\
      .setInputCols("ner_chunk")\
      .setOutputCol("ner_chunk_doc") 

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk_doc"])\
      .setOutputCol("sentence_embeddings")\
      .setCaseSensitive(False)
    
atc_resolver = SentenceEntityResolverModel.load("sbiobertresolve_atc")\
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("atc_code")\
      .setDistanceFunction("EUCLIDEAN")
    
resolver_pipeline = Pipeline(
    stages = [
        document_assembler,
        sentenceDetectorDL,
        tokenizer,
        word_embeddings,
        posology_ner,
        ner_converter,
        c2doc,
        sbert_embedder,
        atc_resolver
  ])

sampleText = ["""He was seen by the endocrinology service and she was discharged on eltrombopag at night, amlodipine with meals metformin two times a day and then ibuprofen.""",
              """She was immediately given hydrogen peroxide 30 mg and amoxicillin twice daily for 10 days to treat the infection on her leg. She has a history of taking magnesium hydroxide.""",
              """She was given antidepressant for a month"""]

data = spark.createDataFrame(pd.DataFrame({"text":sampleText}))

results = resolver_pipeline.fit(data).transform(data)


```
```scala
val document_assembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", 'clinical/models')
      .setInputCols("document")
      .setOutputCol("sentence")

val tokenizer = Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
      .setInputCols("sentence", "token")
      .setOutputCol("word_embeddings")

val posology_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models")
      .setInputCols(Array("sentence", "token", "word_embeddings"))
      .setOutputCol("ner")

val ner_converter = NerConverterInternal()
      .setInputCols(Array("sentence", "token", "ner"))
      .setOutputCol("ner_chunk")
      .setWhiteList(Array('DRUG'))

val c2doc = Chunk2Doc()
      .setInputCols("ner_chunk")
      .setOutputCol("ner_chunk_doc") 

val sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')
      .setInputCols("ner_chunk_doc")
      .setOutputCol("sentence_embeddings")
      .setCaseSensitive(False)
    
val atc_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_atc")
      .setInputCols(Array("ner_chunk", "sentence_embeddings"))
      .setOutputCol("atc_code")
      .setDistanceFunction("EUCLIDEAN")
    

val resolver_pipeline = new PipelineModel().setStages(Array(document_assembler, sentenceDetectorDL, tokenizer, word_embeddings, posology_ner, 
           ner_converter,  c2doc, sbert_embedder, atc_resolver))



val data = Seq("He was seen by the endocrinology service and she was discharged on eltrombopag at night, amlodipine with meals metformin two times a day and then ibuprofen. She was immediately given hydrogen peroxide 30 mg and amoxicillin twice daily for 10 days to treat the infection on her leg. She has a history of taking magnesium hydroxide. She was given antidepressant for a month ")

val results = resolver_pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
+-------------------+--------+--------------------------------------------------+--------------------------------------------------+----------------------------------------+
|              chunk|atc_code|                                       all_k_codes|                                       resolutions|                        all_k_aux_labels|
+-------------------+--------+--------------------------------------------------+--------------------------------------------------+----------------------------------------+
|        eltrombopag| B02BX05|B02BX05:::N06DA05:::N07XX10:::A08AA06:::N06AB09...|eltrombopag; oral:::ipidacrine:::laquinimod:::e...|ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th...|
|         amlodipine| C08CA01|C08CA01:::C08CA17:::C08CA13:::C08CA10:::C07FB12...|amlodipine; oral:::levamlodipine; oral:::lercan...|ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th...|
|          metformin| A10BA02|A10BA02:::V08AA02:::A10BA01:::V08AB01:::P02BB01...|metformin; oral:::metrizoic acid:::phenformin; ...|ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th...|
|          ibuprofen| R02AX02|R02AX02:::M02AA13:::C01EB16:::M01AE13:::M01AE15...|ibuprofen; oral:::ibuprofen; topical:::ibuprofe...|ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th...|
|  hydrogen peroxide| S02AA06|S02AA06:::D10AE:::A02AA03:::C05CA05:::D11AX25::...|hydrogen peroxide; otic:::Peroxides:::magnesium...|ATC 5th:::ATC 4th:::ATC 5th:::ATC 5th...|
|        amoxicillin| J01CA04|J01CA04:::D01AA:::J04AB:::C05AB:::S01AA:::J02AA...|amoxicillin; systemic:::Antibiotics:::Antibioti...|ATC 5th:::ATC 4th:::ATC 4th:::ATC 4th...|
|magnesium hydroxide| A02AA04|A02AA04:::A02AA03:::A12CC:::A02AA:::A06AD04:::G...|magnesium hydroxide; oral (magnesium compounds)...|ATC 5th:::ATC 5th:::ATC 4th:::ATC 4th...|
|     antidepressant|    N06A|N06A:::N05A:::N06AX:::N06D:::N06CA:::D05:::N04A...|ANTIDEPRESSANTS:::ANTIPSYCHOTICS:::Other antide...|ATC 3rd:::ATC 3rd:::ATC 4th:::ATC 3rd...|
+-------------------+--------+--------------------------------------------------+--------------------------------------------------+----------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_atc|
|Compatibility:|Spark NLP 3.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[atc_code]|
|Language:|en|
|Size:|18.6 MB|
|Case sensitive:|false|