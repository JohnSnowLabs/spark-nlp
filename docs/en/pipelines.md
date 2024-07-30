---
layout: docs
header: true
seotitle: Spark NLP - Pipelines
title: Spark NLP - Pipelines
permalink: /docs/en/pipelines
key: docs-pipelines
modify_date: "2024-07-04"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## Pipelines and Models

### Pipelines

**Quick example:**

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq(
  (1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
  (2, "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States")
)).toDF("id", "text")

val pipeline = PretrainedPipeline("explain_document_dl", lang = "en")

val annotation = pipeline.transform(testData)

annotation.show()
/*
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
2.5.0
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
```

</div><div class="h3-box" markdown="1">

#### Showing Available Pipelines

There are functions in Spark NLP that will list all the available Pipelines
of a particular language for you:

```scala
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader

ResourceDownloader.showPublicPipelines(lang = "en")
/*
+--------------------------------------------+------+---------+
| Pipeline                                   | lang | version |
+--------------------------------------------+------+---------+
| dependency_parse                           |  en  | 2.0.2   |
| analyze_sentiment_ml                       |  en  | 2.0.2   |
| check_spelling                             |  en  | 2.1.0   |
| match_datetime                             |  en  | 2.1.0   |
                               ...
| explain_document_ml                        |  en  | 3.1.3   |
+--------------------------------------------+------+---------+
*/
```

Or if we want to check for a particular version:

```scala
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader

ResourceDownloader.showPublicPipelines(lang = "en", version = "3.1.0")
/*
+---------------------------------------+------+---------+
| Pipeline                              | lang | version |
+---------------------------------------+------+---------+
| dependency_parse                      |  en  | 2.0.2   |
                               ...
| clean_slang                           |  en  | 3.0.0   |
| clean_pattern                         |  en  | 3.0.0   |
| check_spelling                        |  en  | 3.0.0   |
| dependency_parse                      |  en  | 3.0.0   |
+---------------------------------------+------+---------+
*/
```

</div><div class="h3-box" markdown="1">

#### Please check out our Models Hub for the full list of [pre-trained pipelines](https://sparknlp.org/models) with examples, demos, benchmarks, and more

### Models

**Some selected languages:
** `Afrikaans, Arabic, Armenian, Basque, Bengali, Breton, Bulgarian, Catalan, Czech, Dutch, English, Esperanto, Finnish, French, Galician, German, Greek, Hausa, Hebrew, Hindi, Hungarian, Indonesian, Irish, Italian, Japanese, Latin, Latvian, Marathi, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Somali, Southern Sotho, Spanish, Swahili, Swedish, Tswana, Turkish, Ukrainian, Zulu`

**Quick online example:**

```python
# load NER model trained by deep learning approach and GloVe word embeddings
ner_dl = NerDLModel.pretrained('ner_dl')
# load NER model trained by deep learning approach and BERT word embeddings
ner_bert = NerDLModel.pretrained('ner_dl_bert')
```

```scala
// load French POS tagger model trained by Universal Dependencies
val french_pos = PerceptronModel.pretrained("pos_ud_gsd", lang = "fr")
// load Italian LemmatizerModel
val italian_lemma = LemmatizerModel.pretrained("lemma_dxc", lang = "it")
````

**Quick offline example:**

- Loading `PerceptronModel` annotator model inside Spark NLP Pipeline

```scala
val french_pos = PerceptronModel.load("/tmp/pos_ud_gsd_fr_2.0.2_2.4_1556531457346/")
  .setInputCols("document", "token")
  .setOutputCol("pos")
```

</div><div class="h3-box" markdown="1">

#### Showing Available Models

There are functions in Spark NLP that will list all the available Models
of a particular Annotator and language for you:

```scala
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader

ResourceDownloader.showPublicModels(annotator = "NerDLModel", lang = "en")
/*
+---------------------------------------------+------+---------+
| Model                                       | lang | version |
+---------------------------------------------+------+---------+
| onto_100                                    |  en  | 2.1.0   |
| onto_300                                    |  en  | 2.1.0   |
| ner_dl_bert                                 |  en  | 2.2.0   |
| onto_100                                    |  en  | 2.4.0   |
| ner_conll_elmo                              |  en  | 3.2.2   |
+---------------------------------------------+------+---------+
*/
```

Or if we want to check for a particular version:

```scala
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader

ResourceDownloader.showPublicModels(annotator = "NerDLModel", lang = "en", version = "3.1.0")
/*
+----------------------------+------+---------+
| Model                      | lang | version |
+----------------------------+------+---------+
| onto_100                   |  en  | 2.1.0   |
| ner_aspect_based_sentiment |  en  | 2.6.2   |
| ner_weibo_glove_840B_300d  |  en  | 2.6.2   |
| nerdl_atis_840b_300d       |  en  | 2.7.1   |
| nerdl_snips_100d           |  en  | 2.7.3   |
+----------------------------+------+---------+
*/
```

And to see a list of available annotators, you can use:

```scala
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader

ResourceDownloader.showAvailableAnnotators()
/*
AlbertEmbeddings
AlbertForTokenClassification
AssertionDLModel
...
XlmRoBertaSentenceEmbeddings
XlnetEmbeddings
*/
```

#### Please check out our Models Hub for the full list of [pre-trained models](https://sparknlp.org/models) with examples, demo, benchmark, and more

</div>