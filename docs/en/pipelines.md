---
layout: article
title: Pipelines
permalink: /docs/en/pipelines
key: docs-pipelines
modify_date: "2019-07-22"
---

## English

**NOTE:**
`noncontrib` pipelines are compatible with `Windows` operating systems.

| Pipelines            | Name                   | en                                                                                                                  |
| -------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| [Explain Document ML](#explain_document_ml)  | `explain_document_ml`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_ml_en_2.1.0_2.4_1563203154682.zip)  |
| [Explain Document DL](#explain_document_dl)  | `explain_document_dl`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_2.1.0_2.4_1562946191575.zip)  |
| Explain Document DL Win | `explain_document_dl_noncontrib`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_noncontrib_en_2.1.0_2.4_1562953993338.zip)  |
| Explain Document DL Fast | `explain_document_dl_fast`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_fast_en_2.1.0_2.4_1562946519404.zip)  |
| Explain Document DL Fast Win | `explain_document_dl_fast_noncontrib`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_fast_noncontrib_en_2.1.0_2.4_1562954323015.zip)  |
| [Recognize Entities DL](#recognize_entities_dl) | `recognize_entities_dl` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/recognize_entities_dl_en_2.1.0_2.4_1562946909722.zip) |
| Recognize Entities DL Win | `recognize_entities_dl_noncontrib` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/recognize_entities_dl_noncontrib_en_2.1.0_2.4_1562954722690.zip) |
| [OntoNotes Entities Small](#onto_recognize_entities_sm) | `onto_recognize_entities_sm` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_recognize_entities_sm_en_2.1.0_2.4_1564330931782.zip)
| [OntoNotes Entities Large](#onto_recognize_entities_lg) | `onto_recognize_entities_lg` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_300_en_2.1.0_2.4_1564256072129.zip)
| [Match Datetime](#match_datetime) | `match_datetime` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/match_datetime_en_2.1.0_2.4_1562944300214.zip)
| [Match Pattern](#match_pattern) | `match_pattern` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/match_pattern_en_2.1.0_2.4_1562944301080.zip)
| [Match Chunk](#match_chunk) | `match_chunk` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/match_chunks_en_2.1.0_2.4_1564415802326.zip)
| Match Phrases | `match_phrases`| [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/match_phrases_en_2.1.0_2.4_1562944304428.zip)
| Clean Stop | `clean_stop`| [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clean_stop_en_2.1.0_2.4_1562944303490.zip)
| Clean Pattern | `clean_pattern`| [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clean_pattern_en_2.1.0_2.4_1562944302303.zip)
| Clean Slang | `clean_slang`| [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clean_slang_en_2.1.0_2.4_1562944852594.zip)
| Check Spelling | `check_spelling`| [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/check_spelling_en_2.1.0_2.4_1562944297070.zip)
| Analyze Sentiment | `analyze_sentiment` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/analyze_sentiment_en_2.1.0_2.4_1563204637489.zip)
| Dependency Parse | `dependency_parse` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_parse_en_2.1.0_2.4_1563224147733.zip)

### explain_document_ml

{% highlight scala %}
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
(2, "The Paris metro will soon enter the 21st century, ditching single-use paper tickets for rechargeable electronic cards.")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("explain_document_ml", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()

/*
2.0.8
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_ml,en,public/models)
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 7 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|            sentence|               token|             checked|              lemmas|               stems|                 pos|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|Google has announ...|[[document, 0, 10...|[[document, 0, 10...|[[token, 0, 5, Go...|[[token, 0, 5, Go...|[[token, 0, 5, Go...|[[token, 0, 5, go...|[[pos, 0, 5, NNP,...|
|  2|The Paris metro w...|[[document, 0, 11...|[[document, 0, 11...|[[token, 0, 2, Th...|[[token, 0, 2, Th...|[[token, 0, 2, Th...|[[token, 0, 2, th...|[[pos, 0, 2, DT, ...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

{% endhighlight %}

### explain_document_dl

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
(2, "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("explain_document_dl", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()
/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_dl,en,public/models)
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 10 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|            sentence|             checked|               lemma|                stem|                 pos|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|Google has announ...|[[document, 0, 10...|[[token, 0, 5, Go...|[[document, 0, 10...|[[token, 0, 5, Go...|[[token, 0, 5, Go...|[[token, 0, 5, go...|[[pos, 0, 5, NNP,...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 5, Go...|
|  2|The Paris metro w...|[[document, 0, 11...|[[token, 0, 2, Th...|[[document, 0, 11...|[[token, 0, 2, Th...|[[token, 0, 2, Th...|[[token, 0, 2, th...|[[pos, 0, 2, DT, ...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 4, 8, Pa...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
+----------------------------------+
|result                            |
+----------------------------------+
|[Google, TensorFlow]              |
|[Donald John Trump, United States]|
+----------------------------------+
*/

{% endhighlight %}

### recognize_entities_dl

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
(2, "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("recognize_entities_dl", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(entity_recognizer_dl,en,public/models)
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 6 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|            sentence|               token|          embeddings|                 ner|       ner_converter|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|Google has announ...|[[document, 0, 10...|[[document, 0, 10...|[[token, 0, 5, Go...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 5, Go...|
|  2|Donald John Trump...|[[document, 0, 92...|[[document, 0, 92...|[[token, 0, 5, Do...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 16, D...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
+----------------------------------+
|result                            |
+----------------------------------+
|[Google, TensorFlow]              |
|[Donald John Trump, United States]|
+----------------------------------+
*/

{% endhighlight %}

### onto_recognize_entities_sm

Trained by **NerDLApproach** annotator with **Char CNN - BiLSTM** and **GloVe Embeddings** on the **OntoNotes** corpus and supports the identification of 18 entities.

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament. "),
(2, "A little less than a decade later, dozens of self-driving startups have cropped up while automakers around the world clamor, wallet in hand, to secure their place in the fast-moving world of fully automated transportation.")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("onto_recognize_entities_sm", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.1.0
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(onto_recognize_entities_sm,en,public/models)
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 6 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|Johnson first ent...|[[document, 0, 17...|[[token, 0, 6, Jo...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 6, Jo...|
|  2|A little less tha...|[[document, 0, 22...|[[token, 0, 0, A,...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 32, A...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
+---------------------------------------------------------------------------------+
|result                                                                           |
+---------------------------------------------------------------------------------+
|[Johnson, first, 2001, Parliament, eight years, London, 2008 to 2016, Parliament]|
|[A little less than a decade later, dozens]                                      |
+---------------------------------------------------------------------------------+
*/

{% endhighlight %}

### onto_recognize_entities_lg

Trained by **NerDLApproach** annotator with **Char CNN - BiLSTM** and **GloVe Embeddings** on the **OntoNotes** corpus and supports the identification of 18 entities.

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament. "),
(2, "A little less than a decade later, dozens of self-driving startups have cropped up while automakers around the world clamor, wallet in hand, to secure their place in the fast-moving world of fully automated transportation.")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("onto_recognize_entities_lg", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.1.0
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(onto_recognize_entities_lg,en,public/models)
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 6 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|Johnson first ent...|[[document, 0, 17...|[[token, 0, 6, Jo...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 6, Jo...|
|  2|A little less tha...|[[document, 0, 22...|[[token, 0, 0, A,...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 32, A...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
+-------------------------------------------------------------------------------+
|result                                                                         |
+-------------------------------------------------------------------------------+
|[Johnson, first, 2001, Parliament, eight years, London, 2008, 2016, Parliament]|
|[A little less than a decade later, dozens]                                    |
+-------------------------------------------------------------------------------+
*/

{% endhighlight %}

### match_datetime

#### DateMatcher yyyy/MM/dd

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "I would like to come over and see you in 01/02/2019."),
(2, "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("match_datetime", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(match_datetime,en,public/models)
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 4 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|            sentence|               token|                date|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|I would like to c...|[[document, 0, 51...|[[document, 0, 51...|[[token, 0, 0, I,...|[[date, 41, 50, 2...|
|  2|Donald John Trump...|[[document, 0, 92...|[[document, 0, 92...|[[token, 0, 5, Do...|[[date, 24, 36, 1...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("date.result").show(false)

/*
+------------+
|result      |
+------------+
|[2019/01/02]|
|[1946/06/14]|
+------------+
*/

{% endhighlight %}

### match_pattern

RegexMatcher (match phone numbers)

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "You should call Mr. Jon Doe at +33 1 79 01 22 89")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("match_pattern", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(match_pattern,en,public/models)
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 4 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|            sentence|               token|               regex|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|You should call M...|[[document, 0, 47...|[[document, 0, 47...|[[token, 0, 2, Yo...|[[chunk, 31, 47, ...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("regex.result").show(false)

/*
+-------------------+
|result             |
+-------------------+
|[+33 1 79 01 22 89]|
+-------------------+
*/

{% endhighlight %}

### match_chunk

The pipeline uses regex <DT/>?/<JJ/>*<NN>+

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "The book has many chapters"),
(2, "the little yellow dog barked at the cat")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("match_chunk", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(match_chunk,en,public/models)
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 5 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|            sentence|               token|                 pos|               chunk|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|The book has many...|[[document, 0, 25...|[[document, 0, 25...|[[token, 0, 2, Th...|[[pos, 0, 2, DT, ...|[[chunk, 0, 7, Th...|
|  2|the little yellow...|[[document, 0, 38...|[[document, 0, 38...|[[token, 0, 2, th...|[[pos, 0, 2, DT, ...|[[chunk, 0, 20, t...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("chunk.result").show(false)

/*
+--------------------------------+
|result                          |
+--------------------------------+
|[The book]                      |
|[the little yellow dog, the cat]|
+--------------------------------+
*/

{% endhighlight %}

## French

| Pipelines               | Name                  | fr                                                                                                                 |
| ----------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [Explain Document Large](#french-explain_document_lg)  | `explain_document_lg` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_fr_2.1.0_2.4_1563178528241.zip) |
| [Explain Document Medium](#french-explain_document_md) | `explain_document_md` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_fr_2.1.0_2.4_1563180522434.zip) |
| [Entity Recognizer Large](#french-entity_recognizer_lg) | `entity_recognizer_lg` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_fr_2.1.0_2.4_1563180776696.zip) |
| [Entity Recognizer Medium](#french-entity_recognizer_md) | `entity_recognizer_md` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_fr_2.1.0_2.4_1563182745366.zip) |

|Feature | Description|
|---|----|
|**NER**|Trained by **NerDLApproach** annotator with **Char CNN - BiLSTM** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities
|**Lemma**|Trained by **Lemmatizer** annotator on **lemmatization-lists** by `Michal Měchura`
|**POS**| Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/fr_gsd/index.html)
|**Size**| Model size indicator, **md** and **lg**. The large pipeline uses **glove_840B_300** and the medium uses **glove_6B_300** WordEmbeddings

### French explain_document_lg

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val pipeline = PretrainedPipeline("explain_document_lg", lang="fr")

val testData = spark.createDataFrame(Seq(
(1, "Contrairement à Quentin Tarantino, le cinéma français ne repart pas les mains vides de la compétition cannoise."),
(2, "Emmanuel Jean-Michel Frédéric Macron est le fils de Jean-Michel Macron, né en 1950, médecin, professeur de neurologie au CHU d'Amiens4 et responsable d'enseignement à la faculté de médecine de cette même ville5, et de Françoise Noguès, médecin conseil à la Sécurité sociale")
)).toDF("id", "text")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_lg,fr,public/models)
testData: org.apache.spark.sql.DataFrame = [id: bigint, text: string]
annotation: org.apache.spark.sql.DataFrame = [id: bigint, text: string ... 8 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|            sentence|               lemma|                 pos|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  0|Contrairement à Q...|[[document, 0, 11...|[[token, 0, 12, C...|[[document, 0, 11...|[[token, 0, 12, C...|[[pos, 0, 12, ADV...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 16, 32, ...|
|  1|Emmanuel Jean-Mic...|[[document, 0, 27...|[[token, 0, 7, Em...|[[document, 0, 27...|[[token, 0, 7, Em...|[[pos, 0, 7, PROP...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 35, E...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*+-------------------------------------------------------------------------------------------------------------+
|result                                                                                                       |
+-------------------------------------------------------------------------------------------------------------+
|[Quentin Tarantino]                                                                                          |
|[Emmanuel Jean-Michel Frédéric Macron, Jean-Michel Macron, CHU d'Amiens4, Françoise Noguès, Sécurité sociale]|
+-------------------------------------------------------------------------------------------------------------+
*/

{% endhighlight %}

### French explain_document_md

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val pipeline = PretrainedPipeline("explain_document_md", lang="fr")

val testData = spark.createDataFrame(Seq(
(1, "Contrairement à Quentin Tarantino, le cinéma français ne repart pas les mains vides de la compétition cannoise."),
(2, "Emmanuel Jean-Michel Frédéric Macron est le fils de Jean-Michel Macron, né en 1950, médecin, professeur de neurologie au CHU d'Amiens4 et responsable d'enseignement à la faculté de médecine de cette même ville5, et de Françoise Noguès, médecin conseil à la Sécurité sociale")
)).toDF("id", "text")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_md,fr,public/models)
testData: org.apache.spark.sql.DataFrame = [id: bigint, text: string]
annotation: org.apache.spark.sql.DataFrame = [id: bigint, text: string ... 8 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|            sentence|               lemma|                 pos|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  0|Contrairement à Q...|[[document, 0, 11...|[[token, 0, 12, C...|[[document, 0, 11...|[[token, 0, 12, C...|[[pos, 0, 12, ADV...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 16, 32, ...|
|  1|Emmanuel Jean-Mic...|[[document, 0, 27...|[[token, 0, 7, Em...|[[document, 0, 27...|[[token, 0, 7, Em...|[[pos, 0, 7, PROP...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 35, E...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
|result                                                                                                          |
+----------------------------------------------------------------------------------------------------------------+
|[Quentin Tarantino]                                                                                             |
|[Emmanuel Jean-Michel Frédéric Macron, Jean-Michel Macron, au CHU d'Amiens4, Françoise Noguès, Sécurité sociale]|
+----------------------------------------------------------------------------------------------------------------+
*/

{% endhighlight %}

### French entity_recognizer_lg

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val pipeline = PretrainedPipeline("entity_recognizer_lg", lang="fr")

val testData = spark.createDataFrame(Seq(
(1, "Contrairement à Quentin Tarantino, le cinéma français ne repart pas les mains vides de la compétition cannoise."),
(2, "Emmanuel Jean-Michel Frédéric Macron est le fils de Jean-Michel Macron, né en 1950, médecin, professeur de neurologie au CHU d'Amiens4 et responsable d'enseignement à la faculté de médecine de cette même ville5, et de Françoise Noguès, médecin conseil à la Sécurité sociale")
)).toDF("id", "text")

val annotation = pipeline.transform(testData)

annotation.show()

/*
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|            sentence|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  0|Contrairement à Q...|[[document, 0, 11...|[[token, 0, 12, C...|[[document, 0, 11...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 16, 32, ...|
|  1|Emmanuel Jean-Mic...|[[document, 0, 27...|[[token, 0, 7, Em...|[[document, 0, 27...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 35, E...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
+-------------------------------------------------------------------------------------------------------------+
|result                                                                                                       |
+-------------------------------------------------------------------------------------------------------------+
|[Quentin Tarantino]                                                                                          |
|[Emmanuel Jean-Michel Frédéric Macron, Jean-Michel Macron, CHU d'Amiens4, Françoise Noguès, Sécurité sociale]|
+-------------------------------------------------------------------------------------------------------------+
*/

{% endhighlight %}

### French entity_recognizer_md

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val pipeline = PretrainedPipeline("entity_recognizer_md", lang="fr")

val testData = spark.createDataFrame(Seq(
(1, "Contrairement à Quentin Tarantino, le cinéma français ne repart pas les mains vides de la compétition cannoise."),
(2, "Emmanuel Jean-Michel Frédéric Macron est le fils de Jean-Michel Macron, né en 1950, médecin, professeur de neurologie au CHU d'Amiens4 et responsable d'enseignement à la faculté de médecine de cette même ville5, et de Françoise Noguès, médecin conseil à la Sécurité sociale")
)).toDF("id", "text")

val annotation = pipeline.transform(testData)

annotation.show()

/*
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|            sentence|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  0|Contrairement à Q...|[[document, 0, 11...|[[token, 0, 12, C...|[[document, 0, 11...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 16, 32, ...|
|  1|Emmanuel Jean-Mic...|[[document, 0, 27...|[[token, 0, 7, Em...|[[document, 0, 27...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 35, E...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*+-------------------------------------------------------------------------------------------------------------+
|result                                                                                                          |
+----------------------------------------------------------------------------------------------------------------+
|[Quentin Tarantino]                                                                                             |
|[Emmanuel Jean-Michel Frédéric Macron, Jean-Michel Macron, au CHU d'Amiens4, Françoise Noguès, Sécurité sociale]|
+----------------------------------------------------------------------------------------------------------------+
*/

{% endhighlight %}

## Italian

| Pipelines               | Name                  | it                                                                                                                 |
| ----------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [Explain Document Large](#italian-explain_document_lg)  | `explain_document_lg`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_it_2.1.0_2.4_1563183013508.zip)  |
| [Explain Document Medium](#italian-explain_document_md) | `explain_document_md`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_it_2.1.0_2.4_1563184262421.zip)  |
| [Entity Recognizer Large](#italian-entity_recognizer_lg) | `entity_recognizer_lg` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_it_2.1.0_2.4_1563184543759.zip) |
| [Entity Recognizer Medium](#italian-entity_recognizer_md) | `entity_recognizer_md` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_it_2.1.0_2.4_1563186026810.zip) |

|Feature | Description|
|---|----|
|**NER**|Trained by **NerDLApproach** annotator with **Char CNN - BiLSTM** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities
|**Lemma**|Trained by **Lemmatizer** annotator on **DXC Technology** dataset
|**POS**| Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/it_isdt/index.html)
|**Size**| Model size indicator, **md** and **lg**. The large pipeline uses **glove_840B_300** and the medium uses **glove_6B_300** WordEmbeddings

### Italian explain_document_lg

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val pipeline = PretrainedPipeline("explain_document_lg", lang="it")

val testData = spark.createDataFrame(Seq(
(1, "La FIFA ha deciso: tre giornate a Zidane, due a Materazzi"),
(2, "Reims, 13 giugno 2019 – Domani può essere la giornata decisiva per il passaggio agli ottavi di finale dei Mondiali femminili.")
)).toDF("id", "text")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_lg,it,public/models)
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 8 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|            sentence|               lemma|                 pos|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|La FIFA ha deciso...|[[document, 0, 56...|[[token, 0, 1, La...|[[document, 0, 56...|[[token, 0, 1, La...|[[pos, 0, 1, DET,...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 3, 6, FI...|
|  2|Reims, 13 giugno ...|[[document, 0, 12...|[[token, 0, 4, Re...|[[document, 0, 12...|[[token, 0, 4, Re...|[[pos, 0, 4, PROP...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 4, Re...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
+-----------------------------------+
|result                             |
+-----------------------------------+
|[FIFA, Zidane, Materazzi]          |
|[Reims, Domani, Mondiali femminili]|
+-----------------------------------+
*/

{% endhighlight %}

### Italian explain_document_md

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val pipeline = PretrainedPipeline("explain_document_md", lang="it")

val testData = spark.createDataFrame(Seq(
(1, "La FIFA ha deciso: tre giornate a Zidane, due a Materazzi"),
(2, "Reims, 13 giugno 2019 – Domani può essere la giornata decisiva per il passaggio agli ottavi di finale dei Mondiali femminili.")
)).toDF("id", "text")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_lg,it,public/models)
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 8 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|            sentence|               lemma|                 pos|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|La FIFA ha deciso...|[[document, 0, 56...|[[token, 0, 1, La...|[[document, 0, 56...|[[token, 0, 1, La...|[[pos, 0, 1, DET,...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 9, La...|
|  2|Reims, 13 giugno ...|[[document, 0, 12...|[[token, 0, 4, Re...|[[document, 0, 12...|[[token, 0, 4, Re...|[[pos, 0, 4, PROP...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 4, Re...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
+-------------------------------+
|result                         |
+-------------------------------+
|[La FIFA, Zidane, Materazzi]|
|[Reims, Domani, Mondiali]      |
+-------------------------------+
*/

{% endhighlight %}

### Italian entity_recognizer_lg

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val pipeline = PretrainedPipeline("entity_recognizer_lg", lang="it")

val testData = spark.createDataFrame(Seq(
(1, "La FIFA ha deciso: tre giornate a Zidane, due a Materazzi"),
(2, "Reims, 13 giugno 2019 – Domani può essere la giornata decisiva per il passaggio agli ottavi di finale dei Mondiali femminili.")
)).toDF("id", "text")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_lg,it,public/models)
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 8 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|            sentence|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|La FIFA ha deciso...|[[document, 0, 56...|[[token, 0, 1, La...|[[document, 0, 56...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 3, 6, FI...|
|  2|Reims, 13 giugno ...|[[document, 0, 12...|[[token, 0, 4, Re...|[[document, 0, 12...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 4, Re...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
+-----------------------------------+
|result                             |
+-----------------------------------+
|[FIFA, Zidane, Materazzi]          |
|[Reims, Domani, Mondiali femminili]|
+-----------------------------------+
*/

{% endhighlight %}

### Italian entity_recognizer_md

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val pipeline = PretrainedPipeline("entity_recognizer_md", lang="it")

val testData = spark.createDataFrame(Seq(
(1, "La FIFA ha deciso: tre giornate a Zidane, due a Materazzi"),
(2, "Reims, 13 giugno 2019 – Domani può essere la giornata decisiva per il passaggio agli ottavi di finale dei Mondiali femminili.")
)).toDF("id", "text")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(explain_document_lg,it,public/models)
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
annotation: org.apache.spark.sql.DataFrame = [id: int, text: string ... 8 more fields]
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
| id|                text|            document|               token|            sentence|          embeddings|                 ner|            entities|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|  1|La FIFA ha deciso...|[[document, 0, 56...|[[token, 0, 1, La...|[[document, 0, 56...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 9, La...|
|  2|Reims, 13 giugno ...|[[document, 0, 12...|[[token, 0, 4, Re...|[[document, 0, 12...|[[word_embeddings...|[[named_entity, 0...|[[chunk, 0, 4, Re...|
+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
*/

annotation.select("entities.result").show(false)

/*
+-------------------------------+
|result                         |
+-------------------------------+
|[La FIFA, Zidane, Materazzi]|
|[Reims, Domani, Mondiali]      |
+-------------------------------+
*/

{% endhighlight %}

## How to use

### Online

To use Spark NLP pretrained pipelines, you can call `PretrainedPipeline` with pipeline's name and its language (default is `en`):

{% highlight python %}

pipeline = PretrainedPipeline('explain_document_dl', lang='en')

{% endhighlight %}

Same in Scala

{% highlight scala %}

val pipeline = PretrainedPipeline("explain_document_dl", lang="en")

{% endhighlight %}

### Offline

If you have any trouble using online pipelines or models in your environment (maybe it's air-gapped), you can directly download them for `offline` use.

After downloading offline models/pipelines and extracting them, here is how you can use them iside your code (the path could be a shared storage like HDFS in a cluster):

{% highlight scala %}
val advancedPipeline = PipelineModel.load("/tmp/explain_document_dl_en_2.0.2_2.4_1556530585689/")
// To use the loaded Pipeline for prediction
advancedPipeline.transform(predictionDF)

{% endhighlight %}
