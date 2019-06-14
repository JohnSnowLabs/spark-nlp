---
layout: article
title: Pipelines
permalink: /docs/en/pipelines
key: docs-pipelines
modify_date: "2019-06-13"
---

## English

| Pipelines            | Name                   | en                                                                                                                  |
| -------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| [Explain Document ML](#explain_document_ml)  | `explain_document_ml`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_ml_en_2.0.2_2.4_1556661821108.zip)  |
| [Explain Document DL](#explain_document_dl)  | `explain_document_dl`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_2.0.2_2.4_1556530585689.zip)  |
| [Entity Recognizer DL](#entity_recognizer_dl) | `entity_recognizer_dl` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_dl_en_2.0.0_2.4_1553230844671.zip) |
| Analyze Sentiment ML | `analyze_sentiment_ml` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/analyze_sentiment_ml_en_2.0.0_2.4_1553538566020.zip)
| [Dependency Parse](#dependency_parse) | `dependency_parse` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_parse_en_2.0.2_2.4_1559024638093.zip)

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

### entity_recognizer_dl

{% highlight scala %}

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
(1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
(2, "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("entity_recognizer_dl", lang="en")

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

## French

| Pipelines               | Name                  | fr                                                                                                                 |
| ----------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [Explain Document Large](#french-explain_document_lg)  | `explain_document_lg` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_fr_2.0.2_2.4_1559054673712.zip) |
| [Explain Document Medium](#french-explain_document_md) | `explain_document_md` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_fr_2.0.2_2.4_1559118515465.zip) |
| [Entity Recognizer Large](#french-entity_recognizer_lg) | `entity_recognizer_lg` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_fr_2.0.8_2.4_1560367295894.zip) |
| [Entity Recognizer Medium](#french-entity_recognizer_md) | `entity_recognizer_md` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_fr_2.0.8_2.4_1560368417326.zip) |

|Feature | Description|
|---|----|
|**NER**|Trained by **NerDLApproach** annotator with **BiLSTM-CNN** on the **WikiNER** corpus and supports identification of `PER`, `LOC`, `ORG` and `MISC` entities
|**Lemma**|Trained by **Lemmatizer** annotator on **lemmatization-lists** by `Michal Měchura`
|**POS**| Trained by **PerceptronApproach** annotator on the **Universal Dependencies**
|**Size**| Model size indicator, **md** and **lg**. The large pipeline uses **glove_840B_300** and the medium uses **glove_6B_300**

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

`Size`: Model size indicator, `md` and `lg`. The large pipeline uses `glove_840B_300` and the medium uses `glove_6B_300`.

| Pipelines               | Name                  | it                                                                                                                 |
| ----------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [Explain Document Large](#italian-explain_document_lg)  | `explain_document_lg`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_it_2.0.8_2.4_1560346123709.zip)  |
| [Explain Document Medium](#italian-explain_document_md) | `explain_document_md`  | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_it_2.0.8_2.4_1560368705919.zip)  |
| [Entity Recognizer Large](#italian-entity_recognizer_lg) | `entity_recognizer_lg` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_it_2.0.8_2.4_1560368922718.zip) |
| [Entity Recognizer Medium](#italian-entity_recognizer_md) | `entity_recognizer_md` | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_it_2.0.8_2.4_1560370005870.zip) |

|Feature | Description|
|---|----|
|**NER**|Trained by **NerDLApproach** annotator with **BiLSTM-CNN** on the **WikiNER** corpus and supports identification of `PER`, `LOC`, `ORG` and `MISC` entities
|**Lemma**|Trained by **Lemmatizer** annotator on **DXC Technology** dataset`
|**POS**| Trained by **PerceptronApproach** annotator on the **Universal Dependencies**
|**Size**| Model size indicator, **md** and **lg**. The large pipeline uses **glove_840B_300** and the medium uses **glove_6B_300**

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
