---
layout: model
title: Sentence Entity Resolver for ATC (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_atc
date: 2022-03-01
tags: [atc, licensed, en, clinical, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps drugs entities to ATC (Anatomic Therapeutic Chemical) codes using `sbiobert_base_cased_mli ` Sentence Bert Embeddings.

## Predicted Entities

`ATC Codes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_atc_en_3.4.1_3.0_1646126349436.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_atc_en_3.4.1_3.0_1646126349436.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") \
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
      .setWhiteList(["DRUG"])

c2doc = Chunk2Doc()\
      .setInputCols("ner_chunk")\
      .setOutputCol("ner_chunk_doc") 

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
      .setInputCols(["ner_chunk_doc"])\
      .setOutputCol("sentence_embeddings")\
      .setCaseSensitive(False)
    
atc_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_atc", "en", "clinical/models")\
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

sampleText = ["""He was seen by the endocrinology service and she was discharged on eltrombopag at night, amlodipine with meals metformin two times a day.""",
              """She was immediately given hydrogen peroxide 30 mg and amoxicillin twice daily for 10 days to treat the infection on her leg. She has a history of taking magnesium hydroxide.""",
              """She was given antidepressant for a month"""]

data = spark.createDataFrame(sampleText, StringType()).toDF("text")

results = resolver_pipeline.fit(data).transform(data)
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

val posology_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models")
      .setInputCols(Array("sentence", "token", "word_embeddings"))
      .setOutputCol("ner")

val ner_converter = NerConverterInternal()
      .setInputCols(Array("sentence", "token", "ner"))
      .setOutputCol("ner_chunk")
      .setWhiteList(Array("DRUG"))

val c2doc = Chunk2Doc()
      .setInputCols(Array("ner_chunk"))
      .setOutputCol("ner_chunk_doc") 

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
      .setInputCols(Array("ner_chunk_doc"))
      .setOutputCol("sentence_embeddings")
      .setCaseSensitive(False)
    
val atc_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_atc", "en", "clinical/models")
      .setInputCols(Array("ner_chunk", "sentence_embeddings"))
      .setOutputCol("atc_code")
      .setDistanceFunction("EUCLIDEAN")
    

val resolver_pipeline = new PipelineModel().setStages(Array(document_assembler, sentenceDetectorDL, tokenizer, word_embeddings, posology_ner, 
           ner_converter,  c2doc, sbert_embedder, atc_resolver))

val data = Seq("He was seen by the endocrinology service and she was discharged on eltrombopag at night, amlodipine with meals metformin two times a day and then ibuprofen. She was immediately given hydrogen peroxide 30 mg and amoxicillin twice daily for 10 days to treat the infection on her leg. She has a history of taking magnesium hydroxide. She was given antidepressant for a month").toDF("text")

val results = resolver_pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
+-------------------+--------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|              chunk|atc_code|                                       all_k_codes|                                       resolutions|                                  all_k_aux_labels|
+-------------------+--------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|        eltrombopag| B02BX05|B02BX05:::A07DA06:::B06AC03:::M01AB08:::L04AA39...|eltrombopag :::eluxadoline :::ecallantide :::et...|ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th...|
|         amlodipine| C08CA01|C08CA01:::C08CA17:::C08CA13:::C08CA06:::C08CA10...|amlodipine :::levamlodipine :::lercanidipine ::...|ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th...|
|          metformin| A10BA02|A10BA02:::A10BA01:::A10BB01:::A10BH04:::A10BB07...|metformin :::phenformin :::glyburide / metformi...|ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th...|
|  hydrogen peroxide| A01AB02|A01AB02:::S02AA06:::D10AE:::D11AX25:::D10AE01::...|hydrogen peroxide :::hydrogen peroxide; otic:::...|ATC 5th:::ATC 5th:::ATC 4th:::ATC 5th:::ATC 5th...|
|        amoxicillin| J01CA04|J01CA04:::J01CA01:::J01CF02:::J01CF01:::J01CA51...|amoxicillin :::ampicillin :::cloxacillin :::dic...|ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th...|
|magnesium hydroxide| A02AA04|A02AA04:::A12CC02:::D10AX30:::B05XA11:::A02AA02...|magnesium hydroxide :::magnesium sulfate :::alu...|ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th:::ATC 5th...|
|     antidepressant|    N06A|N06A:::N05A:::N06AX:::N05AH02:::N06D:::N06CA:::...|ANTIDEPRESSANTS:::ANTIPSYCHOTICS:::Other antide...|ATC 3rd:::ATC 3rd:::ATC 4th:::ATC 5th:::ATC 3rd...|
+-------------------+--------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_atc|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[atc_code]|
|Language:|en|
|Size:|71.6 MB|
|Case sensitive:|false|

## References

Trained on ATC 2022 Codes dataset
