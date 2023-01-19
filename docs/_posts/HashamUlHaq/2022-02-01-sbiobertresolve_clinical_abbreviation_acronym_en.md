---
layout: model
title: Sentence Entity Resolver for Clinical Abbreviations and Acronyms (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_clinical_abbreviation_acronym
date: 2022-02-01
tags: [en, entity_resolution, clinical, licensed]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.4
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical abbreviations and acronyms to their meanings using `sbiobert_base_cased_mli` Sentence Bert Embeddings. This model is an improved version of the base model, and includes more variational data.

## Predicted Entities

`Abbreviation Meanings`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_clinical_abbreviation_acronym_en_3.3.4_3.0_1643681527227.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_clinical_abbreviation_acronym_en_3.3.4_3.0_1643681527227.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_abbreviation_clinical", "en", "clinical/models") \
    .setInputCols(["document", "token", "word_embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverterInternal() \
    .setInputCols(["document", "token", "ner"]) \
    .setOutputCol("ner_chunk")\
    .setWhiteList(['ABBR'])

sentence_chunk_embeddings = BertSentenceChunkEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
    .setInputCols(["document", "ner_chunk"])\
    .setOutputCol("sentence_embeddings")\
    .setChunkWeight(0.5)\
    .setCaseSensitive(True)

abbr_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_clinical_abbreviation_acronym", "en", "clinical/models") \
    .setInputCols(["ner_chunk", "sentence_embeddings"]) \
    .setOutputCol("abbr_meaning")\
    .setDistanceFunction("EUCLIDEAN")\

resolver_pipeline = Pipeline(
stages = [
document_assembler,
tokenizer,
word_embeddings,
clinical_ner,
ner_converter,
sentence_chunk_embeddings,
abbr_resolver
])

model = resolver_pipeline.fit(spark.createDataFrame([['']]).toDF("text"))

sample_text = "Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA: Laboratory tests include a CBC which is normal. Blood Type: AB positive. Rubella: Immune. VDRL: Nonreactive. Hepatitis C surface antigen: Negative. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet."

abbr_result = model.transform(spark.createDataFrame([[sample_text]]).toDF('text'))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("document", "token"))
    .setOutputCol("word_embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_abbreviation_clinical", "en", "clinical/models") 
    .setInputCols(Array("document", "token", "word_embeddings")) 
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal() 
    .setInputCols(Array("document", "token", "ner")) 
    .setOutputCol("ner_chunk")
    .setWhiteList(Array("ABBR"))

val sentence_chunk_embeddings = BertSentenceChunkEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
    .setInputCols(Array("document", "ner_chunk"))
    .setOutputCol("sentence_embeddings")
    .setChunkWeight(0.5)
    .setCaseSensitive(True)

val abbr_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_clinical_abbreviation_acronym", "en", "clinical/models") 
    .setInputCols(Array("ner_chunk", "sentence_embeddings")) 
    .setOutputCol("abbr_meaning")
    .setDistanceFunction("EUCLIDEAN")

val resolver_pipeline = new Pipeline().setStages(document_assembler, tokenizer, word_embeddings, clinical_ner, ner_converter, sentence_chunk_embeddings, abbr_resolver)

val sample_text = Seq("""Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA: Laboratory tests include a CBC which is normal. Blood Type: AB positive. Rubella: Immune. VDRL: Nonreactive. Hepatitis C surface antigen: Negative. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet.""").toDS().toDF("text")

val abbr_result = resolver_pipeline.fit(sample_text).transform(sample_text)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.clinical_abbreviation_acronym").predict("""Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA: Laboratory tests include a CBC which is normal. Blood Type: AB positive. Rubella: Immune. VDRL: Nonreactive. Hepatitis C surface antigen: Negative. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet.""")
```

</div>

## Results

```bash
|    | chunk   | abbr_meaning                         | all_k_results                                                                                                                                                                                                |
|---:|:--------|:-------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | CBC     | Complete Blood Count                 | Complete Blood Count:::Complete blood count:::blood group in ABO system:::(complement) component 4:::abortion:::carbohydrate antigen:::clear to auscultation:::carcinoembryonic antigen:::cervical (level) 4 |
|  1 | AB      | blood group in ABO system            | blood group in ABO system:::abortion                                                                                                                                                                         |
|  2 | VDRL    | Venereal disease research laboratory | Venereal disease research laboratory:::venous blood gas:::leukocyte esterase:::vertical banded gastroplasty                                                                                                  |
|  3 | HIV     | human immunodeficiency virus         | human immunodeficiency virus:::blood group in ABO system:::abortion:::fluorescent in situ hybridization                                                                                                      |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_clinical_abbreviation_acronym|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[output]|
|Language:|en|
|Size:|112.3 MB|
|Case sensitive:|true|

## References

Trained on in-house curated dataset.