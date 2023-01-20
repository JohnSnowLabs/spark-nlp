---
layout: model
title: Sentence Entity Resolver for UMLS CUI Codes
author: John Snow Labs
name: sbiobertresolve_umls_major_concepts
date: 2021-10-03
tags: [entity_resolution, licensed, clinical, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.2.3
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts to 4 major categories of UMLS CUI codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It has faster load time, with a speedup of about 6X when compared to previous versions.

## Predicted Entities

`This model returns CUI (concept unique identifier) codes for Clinical Findings`, `Medical Devices`, `Anatomical Structures and Injuries & Poisoning terms`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_umls_major_concepts_en_3.2.3_3.0_1633221571574.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_umls_major_concepts_en_3.2.3_3.0_1633221571574.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

```sbiobertresolve_umls_major_concepts``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_jsl``` as NER model. ```Cerebrovascular_Disease, Communicable_Disease, Diabetes, Disease_Syndrome_Disorder, Heart_Disease, Hyperlipidemia, Hypertension, Injury_or_Poisoning, Kidney_Disease, Medical-Device, Obesity, Oncological, Overweight, Psychological_Condition, Symptom, VS_Finding, ImagingFindings, EKG_Findings``` set in ```.setWhiteList()```.


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_umls_major_concepts","en", "clinical/models") \
.setInputCols(["ner_chunk_doc", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

pipeline = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])

data = spark.createDataFrame([["The patient complains of ankle pain after falling from stairs. She has been advised Arthroscopy by her primary care pyhsician"]]).toDF("text")

results = pipeline.fit(data).transform(data)
```
```scala
...
val chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_umls_major_concepts", "en", "clinical/models")
.setInputCols(Array("ner_chunk_doc", "sbert_embeddings")) 
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val p_model = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver))

val data = Seq(""The patient complains of ankle pain after falling from stairs. She has been advised Arthroscopy by her primary care pyhsician"").toDF("text")  

val res = p_model.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.umls").predict("""The patient complains of ankle pain after falling from stairs. She has been advised Arthroscopy by her primary care pyhsician""")
```

</div>

## Results

```bash
|    | ner_chunk                     | code         | code_description                             |
|---:|:------------------------------|:-------------|:---------------------------------------------|
|  0 | ankle                         | C4047548     | bilateral ankle joint pain (finding)         |
|  1 | falling from stairs           | C0417023     | fall from stairs                             |
|  2 | Arthroscopy                   | C0179144     | arthroscope                                  |
|  3 | primary care pyhsician        | C3266804     | referred by primary care physician (finding) |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_umls_major_concepts|
|Compatibility:|Healthcare NLP 3.2.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_chunk_embeddings]|
|Output Labels:|[umls_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on data sampled from https://www.nlm.nih.gov/research/umls/index.html
