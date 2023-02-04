// Databricks notebook source
// Prerequisite: Install the Spark NLP library from maven repo which is compatible to the Databricks Runtime Spark version

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.{BertEmbeddings, SentenceEmbeddings, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.{DocumentAssembler, EmbeddingsFinisher}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel, LSH, Normalizer, SQLTransformer}
import org.apache.spark.ml.feature.{MinHashLSH, MinHashLSHModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

// COMMAND ----------

 var inputTable: String = "test_data"
 var embeddingType: String = "bert"

 /**
Inside databricks create a table test_data with sample dataset.
+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|keyword                 |text1                                    |text2                                                                                                                                                                                                                                    |
+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|laurentian divide       |Music Books                              |Loudspeaker Cabinets Automotive Electrical Parts & Accessories Mouse Pads Cell Phone Cases Sheet Music Music Network Cables Computer Cables Tripods Books Cable Connectors Stringed Instrument Replacement Parts Power Cables Video Games|
|emanuel evidence outline|Books                                    |Books                                                                                                                                                                                                                                    |
|brother copier ink      |Printer Cartridges Printers & All-in-Ones|Printer Cartridges                                                                                                                                                                                                                       |                                                                                                                                                                                                                      |
|manhattan gmat book     |Books Manuals & Guides                   |Books                                                                                                                                                                                                                                    |
|hugo boss blue wallet   |Wallets Perfumes & Colognes              |Wallets Money Clips                                                                                                                                                                                                                      |
+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
 **/

// COMMAND ----------

val df = spark.read.table(inputTable)
val primaryCorpus = df.select("text1").withColumnRenamed("text1","text")
primaryCorpus.show(false)

val secondaryCorpus = df.select("text2").withColumnRenamed("text2","text")
secondaryCorpus.show(false)

// COMMAND ----------

def buildPipeline(): Unit = {
    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setExplodeSentences(false)

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    def embeddings = {
      if("basic".equalsIgnoreCase(embeddingType)) {
        WordEmbeddingsModel.pretrained("glove_100d", "en")
          .setInputCols("sentence", "token")
          .setOutputCol("embeddings")
          .setCaseSensitive(false)
      }else if("bert".equalsIgnoreCase(embeddingType)) {
        BertEmbeddings
          .pretrained("bert_base_cased", "en")
          .setInputCols(Array("sentence", "token"))
          .setOutputCol("embeddings")
          .setCaseSensitive(false)
          .setPoolingLayer(0)
      }else{
        null
      }
    }

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("sentence", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings", "embeddings")
      .setOutputCols("sentence_embeddings_vectors", "embeddings_vectors")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val explodeVectors = new SQLTransformer().setStatement("SELECT EXPLODE(sentence_embeddings_vectors) AS features, * FROM __THIS__")

    val vectorNormalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)
    
    val similartyChecker = new BucketedRandomProjectionLSH().setInputCol("features").setOutputCol("hashes").setBucketLength(6.0).setNumHashTables(6)
  
    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler,
        sentence,
        tokenizer,
        embeddings,
        embeddingsSentence,
        embeddingsFinisher,
        explodeVectors,
        vectorNormalizer,
        similartyChecker))

    val pipelineModel = pipeline.fit(primaryCorpus)
    pipelineModel.write.overwrite().save("/tmp/spark-nlp-model-v1")
  }

// COMMAND ----------



// COMMAND ----------

import org.apache.spark.sql.functions.{col, udf}
val score = udf((s: Long) => (100-s))

// COMMAND ----------

def findSimilarity(): Unit = {
    // load it back in during production
    val similarityCheckingModel = PipelineModel.load("/tmp/spark-nlp-model-v1")
    val primaryDF = similarityCheckingModel.transform(primaryCorpus)
    val dfA = primaryDF.select("text","features","normFeatures").withColumn("rowkey",monotonically_increasing_id())
    //dfA.show()
    val secondaryDF = similarityCheckingModel.transform(secondaryCorpus)
    val dfB = secondaryDF.select("text","features","normFeatures").withColumn("rowkey",monotonically_increasing_id())
    //dfB.show()

    //Feature Transformation
    print("Approximately joining dfA and dfB :")
    // BucketedRandomProjectionLSH
    similarityCheckingModel.stages.last.asInstanceOf[BucketedRandomProjectionLSHModel].approxSimilarityJoin(dfA, dfB, 100)
      .where(col("datasetA.rowkey") === col("datasetB.rowkey"))
      .select(col("datasetA.text").alias("text1"),
        col("datasetB.text").alias("text2"),
        score(col("distCol")).alias("score")).show()
  }


// COMMAND ----------
// create the model only once and use it many times
buildPipeline()

// COMMAND ----------

findSimilarity()

