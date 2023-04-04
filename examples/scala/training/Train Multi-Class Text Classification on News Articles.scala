// Databricks notebook source
// MAGIC %md
// MAGIC ## Train multi-class text classification on news articles
// MAGIC ### Using ClassifierDL, WordEmbeddings, and SentenceEmbeddings

// COMMAND ----------

spark.version

// COMMAND ----------

// print Spark NLP version

import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version

// COMMAND ----------

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import org.apache.spark.ml.Pipeline

// COMMAND ----------

// MAGIC %sh
// MAGIC curl -O 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_train.csv'

// COMMAND ----------

// MAGIC %md
// MAGIC ### Checkout where it saves it

// COMMAND ----------

// MAGIC %fs ls "file:/databricks/driver"

// COMMAND ----------

val trainDataset = spark.read.option("header","true").csv("file:/databricks/driver/news_category_train.csv")


// COMMAND ----------

trainDataset.show
//The content is inside description column
//The label is inside category column

// COMMAND ----------

val documentAssembler = new DocumentAssembler()
   .setInputCol("description")
   .setOutputCol("document")

val token = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("glove_100d", lang = "en")
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
  .setCaseSensitive(false)

//convert word embeddings to sentence embeddings
val sentenceEmbeddings = new SentenceEmbeddings()
  .setInputCols("document", "embeddings")
  .setOutputCol("sentence_embeddings")
  .setStorageRef("glove_100d")

//ClassifierDL accepts SENTENCE_EMBEDDINGS
//UniversalSentenceEncoder or SentenceEmbeddings can produce SENTECE_EMBEDDINGS
val docClassifier = new ClassifierDLApproach()
  .setInputCols("sentence_embeddings")
  .setOutputCol("class")
  .setLabelColumn("category")
  .setBatchSize(64)
  .setMaxEpochs(20)
  .setLr(5e-3f)
  .setDropout(0.5f)

val pipeline = new Pipeline()
  .setStages(
    Array(
      documentAssembler,
      token,
      embeddings,
      sentenceEmbeddings,
      docClassifier
    )
  )

// COMMAND ----------

// Let's train our multi-class classifier
val pipelineModel = pipeline.fit(trainDataset)

// COMMAND ----------

val testDataset = spark.createDataFrame(Seq(
  (0, "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."),
  (1, "Scientists have discovered irregular lumps beneath the icy surface of Jupiter's largest moon, Ganymede. These irregular masses may be rock formations, supported by Ganymede's icy shell for billions of years...")
)).toDF("id", "description")

// COMMAND ----------

testDataset.show

// COMMAND ----------

val prediction = pipelineModel.transform(testDataset)

//actual predicted classes
prediction.select("class.result").show(false)
//metadata related to scores of all classes
prediction.select("class.metadata").show(false)
