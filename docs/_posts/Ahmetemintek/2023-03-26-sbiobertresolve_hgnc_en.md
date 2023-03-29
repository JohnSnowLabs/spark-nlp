---
layout: model
title: Sentence Entity Resolver for HGNC
author: John Snow Labs
name: sbiobertresolve_hgnc
date: 2023-03-26
tags: [hgnc, entity_resolution, clinical, en, licensed]
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

This model maps extracted gene names and their short-form abbreviations to HGNC codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. Also, it returns the locus groups and locus types of the genes as aux labels separated by || under the metadata.

## Predicted Entities

`HGNC Codes`, `Locus Group`, `Locus Type`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_hgnc_en_4.3.2_3.0_1679847290330.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_hgnc_en_4.3.2_3.0_1679847290330.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")


sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols(["sentence"])\
  .setOutputCol("token")


word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence","token"])\
  .setOutputCol("embeddings")


clinical_ner = MedicalNerModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models")\
  .setInputCols(["sentence","token","embeddings"])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
  .setInputCols(["sentence","token","ner"])\
  .setOutputCol("ner_chunk")\
  .setWhiteList(['GENE'])

chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
  .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
  .setInputCols(["ner_chunk_doc"])\
  .setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel\
  .pretrained("sbiobertresolve_hgnc","en", "clinical/models") \
  .setInputCols(["ner_chunk", "sbert_embeddings"]) \
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])


clinical_note = ["Recent studies have suggested a potential link between the double homeobox 4 like 20 (pseudogene), also known as DUX4L20, and FBXO48 and RNA guanine-7 methyltransferase "]


data= spark.createDataFrame([clinical_note]).toDF('text')
results = nlpPipeline.fit(data).transform(data)

```
```scala
val document_assembler = new DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")


val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
  .setInputCols(Array("document"))\
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()\
  .setInputCols(Array("sentence"))\
  .setOutputCol("token")


val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(Array("sentence","token"))\
  .setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models")\
  .setInputCols(Array("sentence","token","embeddings"))\
  .setOutputCol("ner")

val ner_converter = new NerConverter()\
  .setInputCols(Array("sentence","token","ner"))\
  .setOutputCol("ner_chunk")\
  .setWhiteList(Array('GENE'))

val chunk2doc = new Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings\
  .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
  .setInputCols(Array("ner_chunk_doc"))\
  .setOutputCol("sbert_embeddings")

val resolver = SentenceEntityResolverModel\
  .pretrained("sbiobertresolve_hgnc","en", "clinical/models") \
  .setInputCols(Array("ner_chunk", "sbert_embeddings")) \
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus, associated with an acute hepatitis and obesity with a body mass index (BMI) of 33.5 kg/m2").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
---
layout: model
title: Sentence Entity Resolver for HGNC
author: John Snow Labs
name: sbiobertresolve_hgnc
date: 2023-03-26
tags: [hgnc, entity_resolution, clinical, en, licensed]
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

This model maps extracted gene names and their short-form abbreviations to HGNC codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. Also, it returns the locus groups and locus types of the genes as aux labels separated by || under the metadata.

## Predicted Entities

`HGNC Codes`, `Locus Group`, `Locus Type`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_hgnc_en_4.3.2_3.0_1679847290330.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_hgnc_en_4.3.2_3.0_1679847290330.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")


sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols(["sentence"])\
  .setOutputCol("token")


word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence","token"])\
  .setOutputCol("embeddings")


clinical_ner = MedicalNerModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models")\
  .setInputCols(["sentence","token","embeddings"])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
  .setInputCols(["sentence","token","ner"])\
  .setOutputCol("ner_chunk")\
  .setWhiteList(['GENE'])

chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
  .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
  .setInputCols(["ner_chunk_doc"])\
  .setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel\
  .pretrained("sbiobertresolve_hgnc","en", "clinical/models") \
  .setInputCols(["ner_chunk", "sbert_embeddings"]) \
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])


clinical_note = ["Recent studies have suggested a potential link between the double homeobox 4 like 20 (pseudogene), also known as DUX4L20, and FBXO48 and RNA guanine-7 methyltransferase "]


data= spark.createDataFrame([clinical_note]).toDF('text')
results = nlpPipeline.fit(data).transform(data)

```
```scala
val document_assembler = new DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")


val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
  .setInputCols(Array("document"))\
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()\
  .setInputCols(Array("sentence"))\
  .setOutputCol("token")


val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(Array("sentence","token"))\
  .setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models")\
  .setInputCols(Array("sentence","token","embeddings"))\
  .setOutputCol("ner")

val ner_converter = new NerConverter()\
  .setInputCols(Array("sentence","token","ner"))\
  .setOutputCol("ner_chunk")\
  .setWhiteList(Array('GENE'))

val chunk2doc = new Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings\
  .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
  .setInputCols(Array("ner_chunk_doc"))\
  .setOutputCol("sbert_embeddings")

val resolver = SentenceEntityResolverModel\
  .pretrained("sbiobertresolve_hgnc","en", "clinical/models") \
  .setInputCols(Array("ner_chunk", "sbert_embeddings")) \
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus, associated with an acute hepatitis and obesity with a body mass index (BMI) of 33.5 kg/m2").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
|   sent_id | ner_chunk   | entity   | HGNC Code    | all_codes                                                               | resolutions                                                                                                                    | AUX                                                                                                                             |
|----------:|:------------|:---------|:-------------|:------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|
|         0 | DUX4L20     | GENE     | HGNC:50801   | ['HGNC:50801', 'HGNC:31982', 'HGNC:42423', 'HGNC:39776', 'HGNC:42023'...| ['DUX4L20 [double homeobox 4 like 20 (pseudogene)]', 'ANKRD20A4P [ankyrin repeat domain 20 family member A4, pseudogene]', '...| [pseudogene :: pseudogene, pseudogene :: pseudogene, non-coding RNA :: RNA, long non-coding, pseudogene :: pseudogene...|
|         0 | FBXO48      | GENE     | HGNC:33857   | ['HGNC:33857', 'HGNC:4930', 'HGNC:16653', 'HGNC:13114', 'HGNC:23535' ...| ['FBXO48 [F-box protein 48]', 'ZBTB48 [zinc finger and BTB domain containing 48]', 'MRPL48 [mitochondrial ribosomal protein ...| [protein-coding gene :: gene with protein product, protein-coding gene :: gene with protein product, protein-coding gene...|
'''

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_hgnc|
|Compatibility:|Healthcare NLP 4.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[hgnc_code]|
|Language:|en|
|Size:|251.9 MB|
|Case sensitive:|false|

## References

https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_hgnc|
|Compatibility:|Healthcare NLP 4.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[hgnc_code]|
|Language:|en|
|Size:|251.9 MB|
|Case sensitive:|false|

## References

https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/
