---
layout: model
title: Sentence Entity Resolver for Clinical Abbreviations and Acronyms (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_clinical_abbreviation_acronym
date: 2021-12-11
tags: [abbreviation, entity_resolver, licensed, en, clinical, acronym]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical abbreviations and acronyms to their meanings using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It is the first primitive version of abbreviation resolution and will be improved further in the following releases.

## Predicted Entities

`Abbreviation Meanings`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_clinical_abbreviation_acronym_en_3.3.4_2.4_1639224244652.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
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
document_assembler,
tokenizer,
word_embeddings,
clinical_ner,
ner_converter_icd,
entity_extractor,
chunk_merge,
c2doc,
sentence_chunk_embeddings,
abbr_resolver
])

model = resolver_pipeline.fit(spark.createDataFrame([['']]).toDF("text"))

sample_text = "HISTORY OF PRESENT ILLNESS: The patient three weeks ago was seen at another clinic for upper respiratory infection-type symptoms. She was diagnosed with a viral infection and had used OTC medications including Tylenol, Sudafed, and Nyquil."
abbr_result = model.transform(spark.createDataFrame([[text]]).toDF('text'))
```
```scala
...
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


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.clinical_abbreviation_acronym").predict("""HISTORY OF PRESENT ILLNESS: The patient three weeks ago was seen at another clinic for upper respiratory infection-type symptoms. She was diagnosed with a viral infection and had used OTC medications including Tylenol, Sudafed, and Nyquil.""")
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
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[abbr_meaning]|
|Language:|en|
|Size:|104.9 MB|
|Case sensitive:|false|
