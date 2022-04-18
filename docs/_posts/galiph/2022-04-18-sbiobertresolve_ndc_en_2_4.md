---
layout: model
title: Sentence Entity Resolver for NDC (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_ndc
date: 2022-04-18
tags: [ndc, entity_resolution, licensed, en, clinical]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.3.2
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts (like drugs/ingredients) to [National Drug Codes](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory) using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It also returns package options and alternative drugs in the all_k_aux_label column.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_NDC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_ndc_en_3.3.2_2.4_1650298194939.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

c2doc = Chunk2Doc()\
      .setInputCols("ner_chunk")\
      .setOutputCol("ner_chunk_doc") 

sbert_embedder = BertSentenceEmbeddings\
      .pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk_doc"])\
      .setOutputCol("sentence_embeddings")
    
ndc_resolver = SentenceEntityResolverModel\
      .pretrained("sbiobertresolve_ndc", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("ndc")\
      .setDistanceFunction("EUCLIDEAN")\
      .setCaseSensitive(False)
    
resolver_pipeline = Pipeline(
    stages = [
        document_assembler,
        sentenceDetectorDL,
        tokenizer,
        word_embeddings,
        posology_ner,
        ner_converter_icd,
        c2doc,
        sbert_embedder,
        ndc_resolver
  ])

data = spark.createDataFrame([["""text = 'The patient was given aspirin 81 mg and metformin 500 mg'"""]]).toDF("text")

result = resolver_pipeline.fit(data).transform(data)
```
```scala
...

val c2doc = Chunk2Doc()
      .setInputCols("ner_chunk")
      .setOutputCol("ner_chunk_doc") 

val sbert_embedder = BertSentenceEmbeddings
      .pretrained("sbiobert_base_cased_mli", "en","clinical/models")
      .setInputCols(Array("ner_chunk_doc"))
      .setOutputCol("sentence_embeddings")
    
val ndc_resolver = SentenceEntityResolverModel
      .pretrained("sbiobertresolve_ndc", "en", "clinical/models") 
      .setInputCols(Array("ner_chunk", "sentence_embeddings")) 
      .setOutputCol("ndc")
      .setDistanceFunction("EUCLIDEAN")
      .setCaseSensitive(False)
    
val resolver_pipeline = new Pipeline().setStages(Array(
        document_assembler,
        sentenceDetectorDL,
        tokenizer,
        word_embeddings,
        posology_ner,
        ner_converter_icd,
        c2doc,
        sbert_embedder,
        ndc_resolver
        ))

val clinical_note = Seq("The patient was given aspirin 81 mg and metformin 500 mg")

val result = resolver_pipeline.fit(clinical_note).transform(clinical_note)
```
</div>

## Results

```bash
+----------------+----------+----------------------------------------------------------------------------------------------------+
|       ner_chunk|  ndc_code|                                                                                          aux_labels|
+----------------+----------+----------------------------------------------------------------------------------------------------+
|   aspirin 81 mg|41250-0780|{'packages': "['1 BOTTLE, PLASTIC in 1 PACKAGE (41250-780-01)  > 120 TABLET, DELAYED RELEASE in 1...|
|metformin 500 mg|62207-0491|{'packages': "['5000 TABLET in 1 POUCH (62207-491-31)', '25000 TABLET in 1 CARTON (62207-491-35)'...|
+----------------+----------+----------------------------------------------------------------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_ndc|
|Compatibility:|Spark NLP for Healthcare 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[ndc]|
|Language:|en|
|Size:|932.2 MB|
|Case sensitive:|false|

## References

It is trained on U.S. FDA 2022-NDC Codes dataset.
