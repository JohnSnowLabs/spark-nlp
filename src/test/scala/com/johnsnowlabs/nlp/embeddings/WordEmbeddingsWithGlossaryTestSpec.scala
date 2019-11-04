package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer

import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import org.scalatest.FlatSpec
import org.apache.spark.sql.SparkSession


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
     "Ionnidis Papaloudus is a doctor.\n\nHe works in Alabama.",
     "John Smith is a well known name.")
     .toDF("text")


   val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
   val sentenceDetector = new SentenceDetector().setInputCols(Array("document")).setOutputCol("sentence")
   val tokenizer = new Tokenizer().setInputCols(Array("sentence")).setOutputCol("token")

   val glove_100d_embeddings_with_glossary = WordEmbeddingsModel.pretrained("glove_100d", "en")
   //val glove_100d_embeddings_with_glossary = WordEmbeddingsModel.load("use path to glove_100d_en_2.0.2_2.4_1556534397055 for local tests")
   glove_100d_embeddings_with_glossary.setInputCols(Array("sentence", "token")).setOutputCol("embeddings")
   // Now add glossary to glove_100d
   val myVector = Array.fill(100)("1".toFloat) // lets imagine an arbitrary Vector full of 1.0
   val myGlossary: Option[Map[String, Array[Float]]] = Some(Map("Ionnidis" -> myVector, "Papaloudus"-> myVector))
   glove_100d_embeddings_with_glossary.setGlossary(myGlossary)

   val pipeline_with_glossary = new RecursivePipeline()
     .setStages(Array(
       documentAssembler,
       sentenceDetector,
       tokenizer,
       glove_100d_embeddings_with_glossary
     ))

   val myDFWithGlossary = pipeline_with_glossary.fit(srcDF).transform(srcDF)
   myDFWithGlossary.show(false)

 }


}
