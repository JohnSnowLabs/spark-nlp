---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.2.3
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_2_3
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.2.3
We are glad to announce that Spark NLP Healthcare 3.2.3 has been released!.

#### Highlights
+ New BERT-Based Deidentification NER Model
+ New Sentence Entity Resolver Models For German Language
+ New Spell Checker Model For Drugs
+ Allow To Use Disambiguator Pretrained Model
+ Allow To Use Seeds in StructuredDeidentification
+ Added Compatibility with Tensorflow 1.15 For Graph Generation.
+ New Setup Videos

#### New BERT-Based Deidentification NER Model

We have a new `bert_token_classifier_ner_deid` model that is BERT-based version of `ner_deid_subentity_augmented` and annotates text to find protected health information that may need to be de-identified. It can detect 23 different entities (`MEDICALRECORD`, `ORGANIZATION`, `DOCTOR`, `USERNAME`, `PROFESSION`, `HEALTHPLAN`, `URL`, `CITY`, `DATE`, `LOCATION-OTHER`, `STATE`, `PATIENT`, `DEVICE`, `COUNTRY`, `ZIP`, `PHONE`, `HOSPITAL`, `EMAIL`, `IDNUM`, `SREET`, `BIOID`, `FAX`, `AGE`).

*Example*:

```bash
documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

tokenizer = Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_deid", "en")\
  .setInputCols("token", "document")\
  .setOutputCol("ner")\
  .setCaseSensitive(True)

ner_converter = NerConverter()\
  .setInputCols(["document","token","ner"])\
  .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])
p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

text = """A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."""
result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [text]})))
```

*Results*:

```bash
+-----------------------------+-------------+
|chunk                        |ner_label    |
+-----------------------------+-------------+
|2093-01-13                   |DATE         |
|David Hale                   |DOCTOR       |
|Hendrickson, Ora             |PATIENT      |
|7194334                      |MEDICALRECORD|
|Oliveira                     |PATIENT      |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street            |STREET       |
|302) 786-5227                |PHONE        |
|Brothers Coal-Mine           |ORGANIZATION |
+-----------------------------+-------------+
```

#### New Sentence Entity Resolver Models For German Language

We are releasing two new Sentence Entity Resolver Models for German language that use `sent_bert_base_cased` (de) embeddings.

+ `sbertresolve_icd10gm` : This model maps extracted medical entities to ICD10-GM codes for the German language.

*Example*:

```bash
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "de")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("sbert_embeddings")

icd10gm_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_icd10gm", "de", "clinical/models")\
    .setInputCols(["ner_chunk", "sbert_embeddings"])\
    .setOutputCol("icd10gm_code")

icd10gm_pipelineModel = PipelineModel( stages = [documentAssembler, sbert_embedder, icd10gm_resolver])

icd_lp = LightPipeline(icd10gm_pipelineModel)
icd_lp.fullAnnotate("Dyspnoe")
```

*Results* :

|chunk|code|resolutions|all_codes|all_distances|
|-|-|-|-|-|
| Dyspnoe | C671 | Dyspnoe, Schlafapnoe, Dysphonie, Frühsyphilis, Hyperzementose, Hypertrichose, ...  | [R06.0, G47.3, R49.0, A51, K03.4, L68, ...] | [0.0000, 2.5602, 3.0529, 3.3310, 3.4645, 3.7148, ...] |

+ `sbertresolve_snomed` : This model maps extracted medical entities to SNOMED codes for the German language.

*Example*:

```bash
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "de")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("sbert_embeddings")

snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed", "de", "clinical/models")\
    .setInputCols(["ner_chunk", "sbert_embeddings"])\
    .setOutputCol("snomed_code")

snomed_pipelineModel = PipelineModel( stages = [ documentAssembler, sbert_embedder, snomed_resolver])

snomed_lp = LightPipeline(snomed_pipelineModel)
snomed_lp.fullAnnotate("Bronchialkarzinom ")
```

*Results* :

|chunk|code|resolutions|all_codes|all_distances|
|-|-|-|-|-|
| Bronchialkarzinom  | 22628 | Bronchialkarzinom, Bronchuskarzinom, Rektumkarzinom, Klavikulakarzinom, Lippenkarzinom, Urothelkarzinom, ...  | [22628, 111139, 18116, 107569, 18830, 22909, ...] | [0.0000, 0.0073, 0.0090, 0.0098, 0.0098, 0.0102, ...] |

#### New Spell Checker Model For Drugs

We are releasing new `spellcheck_drug_norvig` model that detects and corrects spelling errors of drugs in a text based on the Norvig's approach.

*Example* :

```bash
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer()
    .setInputCols("document")\
    .setOutputCol("token")

spell = NorvigSweetingModel.pretrained("spellcheck_drug_norvig", "en", "clinical/models")\
    .setInputCols("token")
    .setOutputCol("spell")\

pipeline = Pipeline( stages = [documentAssembler,
tokenizer, spell])

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text'))
lp = LightPipeline(model)

lp.annotate("You have to take Neutrcare and colfosrinum and a bit of Fluorometholne & Ribotril")
```

*Results* :

```bash
Original text  : You have to take Neutrcare and colfosrinum and a bit of fluorometholne & Ribotril
Corrected text : You have to take Neutracare and colforsinum and a bit of fluorometholone & Rivotril

```

#### Allow to use Disambiguator pretrained model.

Now we can use the NerDisambiguatorModel as a pretrained model to disambiguate person entities.

```python
 text = "The show also had a contestant named Brad Pitt" \
        + "who later defeated Christina Aguilera on the way to become Female Vocalist Champion in the 1989 edition of Star Search in the United States. "
 data = SparkContextForTest.spark.createDataFrame([
     [text]]) \
     .toDF("text").cache()
 da = DocumentAssembler().setInputCol("text").setOutputCol("document")

 sd = SentenceDetector().setInputCols("document").setOutputCol("sentence")

 tk = Tokenizer().setInputCols("sentence").setOutputCol("token")

 emb = WordEmbeddingsModel.pretrained().setOutputCol("embs")

 semb = SentenceEmbeddings().setInputCols("sentence", "embs").setOutputCol("sentence_embeddings")

 ner = NerDLModel.pretrained().setInputCols("sentence", "token", "embs").setOutputCol("ner")

 nc = NerConverter().setInputCols("sentence", "token", "ner").setOutputCol("ner_chunk").setWhiteList(["PER"])

 NerDisambiguatorModel.pretrained().setInputCols("ner_chunk", "sentence_embeddings").setOutputCol("disambiguation")

 pl = Pipeline().setStages([da, sd, tk, emb, semb, ner, nc, disambiguator])

 data = pl.fit(data).transform(data)
 data.select("disambiguation").show(10, False)

```

```bash
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|disambiguation                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[[disambiguation, 65, 82, http://en.wikipedia.org/?curid=144171, http://en.wikipedia.org/?curid=6636454, [chunk -> Christina Aguilera, titles -> christina aguilera ::::: christina aguilar, links -> http://en.wikipedia.org/?curid=144171 ::::: http://en.wikipedia.org/?curid=6636454, beginInText -> 65, scores -> 0.9764155197864447, 0.9727793647472524, categories -> Musicians, Singers, Actors, Businesspeople, Musicians, Singers, ids -> 144171, 6636454, endInText -> 82], []]]|
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

----------------------

```
#### Allow to use seeds in StructuredDeidentification

Now, we can use a seed for a specific column. The seed is used to randomly select the entities used during obfuscation mode. By providing the same seed, you can replicate the same mapping multiple times.

```python
df = spark.createDataFrame([
            ["12", "12", "Juan García"],
            ["24", "56", "Will Smith"],
            ["56", "32", "Pedro Ximénez"]
        ]).toDF("ID1", "ID2", "NAME")

obfuscator = StructuredDeidentification(spark=spark, columns={"ID1": "ID", "ID2": "ID", "NAME": "PATIENT"},
                                                columnsSeed={"ID1": 23, "ID2": 23},
                                                obfuscateRefSource="faker")
result = obfuscator.obfuscateColumns(df)
result.show(truncate=False)      
```

```bash
+----------+----------+----------------+
|ID1       |ID2       |NAME            |
+----------+----------+----------------+
|[D3379888]|[D3379888]|[Raina Cleaves] |
|[R8448971]|[M8851891]|[Jennell Barre] |
|[M8851891]|[L5448098]|[Norene Salines]|
+----------+----------+----------------+

Here, you can see that as we have provided the same seed `23` for columns `ID1`, and `ID2`, the number `12` which is appears twice in the first row is mapped to the same randomly generated id `D3379888` each time.
```
#### Added compatibility with Tensorflow 1.15 for graph generation
Some users reported problems while using graphs generated by Tensorflow 2.x. We provide compatibility with Tensorflow 1.15 in the `tf_graph_1x` module, that can be used like this,

```
from sparknlp_jsl.training import tf_graph_1x

```

In next releases, we will provide full support for graph generation using Tensorflow 2.x.

#### New Setup Videos

Now we have videos showing how to setup Spark NLP, Spark NLP for Healthcare and Spark OCR on UBUNTU.

+ [How to Setup Spark NLP on UBUNTU](https://www.youtube.com/watch?v=ZnFENM-yNfQ)
+ [How to Setup Spark NLP for HEALTHCARE on UBUNTU](https://www.youtube.com/watch?v=yKnF-_oz0GE)
+ [How to Setup Spark OCR on UBUNTU](https://www.youtube.com/watch?v=cmt4WIcL0nI)

**To see more, please check**: [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_2_2">Version 3.2.2</a>
    </li>
    <li>
        <strong>Version 3.2.3</strong>
    </li>
    <li>
        <a href="release_notes_3_3_0">Version 3.3.0</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_1">3.5.1</a></li>
    <li><a href="release_notes_3_5_0">3.5.0</a></li>
    <li><a href="release_notes_3_4_2">3.4.2</a></li>
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
    <li><a href="release_notes_3_3_2">3.3.2</a></li>
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li class="active"><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_3">3.1.3</a></li>
    <li><a href="release_notes_3_1_2">3.1.2</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li><a href="release_notes_3_1_0">3.1.0</a></li>
    <li><a href="release_notes_3_0_3">3.0.3</a></li>
    <li><a href="release_notes_3_0_2">3.0.2</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li><a href="release_notes_2_7_6">2.7.6</a></li>
    <li><a href="release_notes_2_7_5">2.7.5</a></li>
    <li><a href="release_notes_2_7_4">2.7.4</a></li>
    <li><a href="release_notes_2_7_3">2.7.3</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
    <li><a href="release_notes_2_7_0">2.7.0</a></li>
    <li><a href="release_notes_2_6_2">2.6.2</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>