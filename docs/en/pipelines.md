---
layout: docs
header: true
seotitle: Spark NLP - Pipelines
title: Spark NLP - Pipelines
permalink: /docs/en/pipelines
key: docs-pipelines
modify_date: "2021-11-20"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

Pretrained Pipelines have moved to Models Hub.
Please follow this link for the updated list of all models and pipelines:
[Models Hub](https://sparknlp.org/modelss)
{:.success}
{:.success}

</div><div class="h3-box" markdown="1">

## English

**NOTE:**
`noncontrib` pipelines are compatible with `Windows` operating systems.

{:.table-model-big}
| Pipelines            | Name                   |
| -------------------- | ---------------------- |
| [Explain Document ML](#explaindocumentml)  | `explain_document_ml`
| [Explain Document DL](#explaindocumentdl)  | `explain_document_dl`
| [Explain Document DL Win]() | `explain_document_dl_noncontrib`
| Explain Document DL Fast | `explain_document_dl_fast`
| Explain Document DL Fast Win | `explain_document_dl_fast_noncontrib`  |
| [Recognize Entities DL](#recognizeentitiesdl) | `recognize_entities_dl` |
| Recognize Entities DL Win | `recognize_entities_dl_noncontrib` |
| [OntoNotes Entities Small](#ontorecognizeentitiessm) | `onto_recognize_entities_sm` |
| [OntoNotes Entities Large](#ontorecognizeentitieslg) | `onto_recognize_entities_lg` |
| [Match Datetime](#matchdatetime) | `match_datetime` |
| [Match Pattern](#matchpattern) | `match_pattern` |
| [Match Chunk](#matchchunks) | `match_chunks` |
| Match Phrases | `match_phrases`|
| Clean Stop | `clean_stop`|
| Clean Pattern | `clean_pattern`|
| Clean Slang | `clean_slang`|
| Check Spelling | `check_spelling`|
| Analyze Sentiment | `analyze_sentiment` |
| Analyze Sentiment DL | `analyze_sentimentdl_use_imdb` |
| Analyze Sentiment DL | `analyze_sentimentdl_use_twitter` |
| Dependency Parse | `dependency_parse` |

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

### onto_recognize_entities_sm

Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **OntoNotes** corpus and supports the identification of 18 entities.

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

</div><div class="h3-box" markdown="1">

### onto_recognize_entities_lg

Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **OntoNotes** corpus and supports the identification of 18 entities.

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

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

### match_chunks

The pipeline uses regex `<DT/>?/<JJ/>*<NN>+`

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "The book has many chapters"),
(2, "the little yellow dog barked at the cat")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("match_chunks", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()

/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.0.8
testData: org.apache.spark.sql.DataFrame = [id: int, text: string]
pipeline: com.johnsnowlabs.nlp.pretrained.PretrainedPipeline = PretrainedPipeline(match_chunks,en,public/models)
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

</div><div class="h3-box" markdown="1">

## French

{:.table-model-big}
| Pipelines               | Name                  |
| ----------------------- | --------------------- |
| [Explain Document Large](#french-explain_document_lg)  | `explain_document_lg` |
| [Explain Document Medium](#french-explain_document_md) | `explain_document_md` |
| [Entity Recognizer Large](#french-entity_recognizer_lg) | `entity_recognizer_lg` |
| [Entity Recognizer Medium](#french-entity_recognizer_md) | `entity_recognizer_md` |

{:.table-model-big}
|Feature | Description|
|---|----|
|**NER**|Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities
|**Lemma**|Trained by **Lemmatizer** annotator on **lemmatization-lists** by `Michal Měchura`
|**POS**| Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/fr_gsd/index.html)
|**Size**| Model size indicator, **md** and **lg**. The large pipeline uses **glove_840B_300** and the medium uses **glove_6B_300** WordEmbeddings

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

## Italian

{:.table-model-big}
| Pipelines               | Name                  |
| ----------------------- | --------------------- |
| [Explain Document Large](#italian-explain_document_lg)  | `explain_document_lg`  |
| [Explain Document Medium](#italian-explain_document_md) | `explain_document_md`  |
| [Entity Recognizer Large](#italian-entity_recognizer_lg) | `entity_recognizer_lg` |
| [Entity Recognizer Medium](#italian-entity_recognizer_md) | `entity_recognizer_md` |

{:.table-model-big}
|Feature | Description|
|---|----|
|**NER**|Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities
|**Lemma**|Trained by **Lemmatizer** annotator on **DXC Technology** dataset
|**POS**| Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/it_isdt/index.html)
|**Size**| Model size indicator, **md** and **lg**. The large pipeline uses **glove_840B_300** and the medium uses **glove_6B_300** WordEmbeddings

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

## Spanish

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.4.0 |   `es` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_es_2.4.0_2.4_1581977077084.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.4.0 |   `es` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_es_2.4.0_2.4_1581976836224.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.4.0 |   `es` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_es_2.4.0_2.4_1581975536033.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.4.0 |   `es` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_es_2.4.0_2.4_1581978479912.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.4.0 |   `es` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_es_2.4.0_2.4_1581978260094.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.4.0 |   `es` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_es_2.4.0_2.4_1581977172660.zip)  |

{:.table-model-big}
| Feature   | Description                                                                                                                                                                                            |
|:----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Lemma** | Trained by **Lemmatizer** annotator on **lemmatization-lists** by `Michal Měchura`                                                                                                                     |
| **POS**   | Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/es_gsd/index.html)                                                             |
| **NER**   | Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities |
|**Size**| Model size indicator, **sm**, **md**, and **lg**. The small pipelines use **glove_100d**, the medium pipelines use **glove_6B_300**, and large pipelines use **glove_840B_300** WordEmbeddings

</div><div class="h3-box" markdown="1">

## Russian

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.4.4 |   `ru` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_ru_2.4.4_2.4_1584017142719.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.4.4 |   `ru` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_ru_2.4.4_2.4_1584016917220.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.4.4 |   `ru` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_ru_2.4.4_2.4_1584015824836.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.4.4 |   `ru` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_ru_2.4.4_2.4_1584018543619.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.4.4 |   `ru` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_ru_2.4.4_2.4_1584018332357.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.4.4 |   `ru` |             | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_ru_2.4.4_2.4_1584017227871.zip)  |

{:.table-model-big}
| Feature   | Description                                                                                                                                                                                            |
|:----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Lemma** | Trained by **Lemmatizer** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/ru_gsd/index.html)|
| **POS**   | Trained by **PerceptronApproach** annotator on the [Universal Dependencies](https://universaldependencies.org/treebanks/ru_gsd/index.html)                                                             |
| **NER**   | Trained by **NerDLApproach** annotator with **Char CNNs - BiLSTM - CRF** and **GloVe Embeddings** on the **WikiNER** corpus and supports the identification of `PER`, `LOC`, `ORG` and `MISC` entities |

</div><div class="h3-box" markdown="1">

## Dutch

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `nl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_nl_2.5.0_2.4_1588546621618.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `nl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_nl_2.5.0_2.4_1588546605329.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `nl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_nl_2.5.0_2.4_1588612556770.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `nl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_nl_2.5.0_2.4_1588546655907.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `nl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_nl_2.5.0_2.4_1588546645304.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `nl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_nl_2.5.0_2.4_1588612569958.zip)  |

</div><div class="h3-box" markdown="1">

## Norwegian

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `no` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_no_2.5.0_2.4_1588784132955.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `no` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_no_2.5.0_2.4_1588783879809.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `no` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_no_2.5.0_2.4_1588782610672.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `no` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_no_2.5.0_2.4_1588794567766.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `no` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_no_2.5.0_2.4_1588794357614.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `no` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_no_2.5.0_2.4_1588793261642.zip)  |

</div><div class="h3-box" markdown="1">

## Polish

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `pl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_pl_2.5.0_2.4_1588531081173.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `pl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_pl_2.5.0_2.4_1588530841737.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `pl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_pl_2.5.0_2.4_1588529695577.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `pl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_pl_2.5.0_2.4_1588532616080.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `pl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_pl_2.5.0_2.4_1588532376753.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `pl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_pl_2.5.0_2.4_1588531171903.zip)  |

</div><div class="h3-box" markdown="1">

## Portuguese

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `pt` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_pt_2.5.0_2.4_1588501423743.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `pt` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_pt_2.5.0_2.4_1588501189804.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `pt` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_pt_2.5.0_2.4_1588500056427.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `pt` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_pt_2.5.0_2.4_1588502815900.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `pt` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_pt_2.5.0_2.4_1588502606198.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `pt` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_pt_2.5.0_2.4_1588501526324.zip)  |

</div><div class="h3-box" markdown="1">

## Multi-language

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| LanguageDetectorDL    | `detect_language_7`        | 2.5.2 |      `xx` |        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/detect_language_7_xx_2.5.0_2.4_1591875676774.zip) |
| LanguageDetectorDL    | `detect_language_20`        | 2.5.2 |      `xx` |        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/detect_language_20_xx_2.5.0_2.4_1591875683182.zip) |

* The model with 7 languages: Czech, German, English, Spanish, French, Italy, and Slovak
* The model with 20 languages: Bulgarian, Czech, German, Greek, English, Spanish, Finnish, French, Croatian, Hungarian, Italy, Norwegian, Polish, Portuguese, Romanian, Russian, Slovak, Swedish, Turkish, and Ukrainian

</div><div class="h3-box" markdown="1">

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

</div><div class="h3-box" markdown="1">

### Offline

If you have any trouble using online pipelines or models in your environment (maybe it's air-gapped), you can directly download them for `offline` use.

After downloading offline models/pipelines and extracting them, here is how you can use them iside your code (the path could be a shared storage like HDFS in a cluster):

{% highlight scala %}
val advancedPipeline = PipelineModel.load("/tmp/explain_document_dl_en_2.0.2_2.4_1556530585689/")
// To use the loaded Pipeline for prediction
advancedPipeline.transform(predictionDF)

{% endhighlight %}

</div>