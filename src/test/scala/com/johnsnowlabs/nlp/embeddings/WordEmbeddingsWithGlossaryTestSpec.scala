package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotator.{NerConverter, NerDLModel, SentenceDetector}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsFormat, WordEmbeddingsModel}
import org.scalatest.FlatSpec
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.util.PipelineModels


class WordEmbeddingsWithGlossaryTestSpec extends FlatSpec {

 "WordEmbeddings" should "force vectors from a glossary" in {

   val spark: SparkSession = SparkSession
     .builder()
     .appName("test")
     .master("local")
     .config("spark.driver.memory", "4G")
     .getOrCreate()

   import spark.implicits._

   val srcDF = Seq(
     "John Smith",
     "Ionnidis Papaloudus")
     .toDF("text")

   val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
   val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
   // first pipeline without glossary so "Ionnidis Papaloudus will not be in the embeddings.
   //val glove_100d_embeddings = WordEmbeddingsModel.load("/sandbox/jsl/cache_pretrained/glove_100d_en_2.0.2_2.4_1556534397055")
   val glove_100d_embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")
   glove_100d_embeddings.setInputCols(Array("document", "token")).setOutputCol("embeddings")

   val pipeline = new RecursivePipeline()
     .setStages(Array(
       documentAssembler,
       tokenizer,
       glove_100d_embeddings
     ))

   val myDF: DataFrame = pipeline.fit(srcDF).transform(srcDF)
   myDF.show(false)

   //val glove_100d_embeddings_with_glossary = WordEmbeddingsModel.load("/sandbox/jsl/cache_pretrained/glove_100d_en_2.0.2_2.4_1556534397055")
   val glove_100d_embeddings_with_glossary = WordEmbeddingsModel.pretrained("glove_100d", "en")
   glove_100d_embeddings_with_glossary.setInputCols(Array("document", "token")).setOutputCol("embeddings")
   // Now add glossary to glove_100d
   val myVector = Array.fill(100)("1".toFloat) // lets imagine an arbitrary Vector full of 1.0
   val myGlossary: Option[Map[String, Array[Float]]] = Some(Map("Ionnidis" -> myVector, "Papaloudus"-> myVector))
   glove_100d_embeddings_with_glossary.setGlossary(myGlossary)

   val pipeline_with_glossary = new RecursivePipeline()
     .setStages(Array(
       documentAssembler,
       tokenizer,
       glove_100d_embeddings_with_glossary
     ))

   val myDFWithGlossary = pipeline_with_glossary.fit(srcDF).transform(srcDF)
   myDFWithGlossary.show(false)

 }


}
