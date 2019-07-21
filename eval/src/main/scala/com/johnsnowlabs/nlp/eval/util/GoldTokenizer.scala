package com.johnsnowlabs.nlp.eval.util

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DocumentAssembler}
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.spark.sql.{Column, Dataset, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.MetadataBuilder

import scala.collection.mutable.ArrayBuffer

class GoldTokenizer(sparkSession: SparkSession) {

  private val TOKEN_POSITION = 0
  private val POS_POSITION = 1
  private val TAG_POSITION = 3

  private def customeTokenizer: UserDefinedFunction = udf { (tokens: Seq[String], text: String, sentenceIndex: String) =>

    val tokenTagAnnotation: ArrayBuffer[Annotation] = ArrayBuffer()
    def annotatorType: String = AnnotatorType.TOKEN
    var lastIndex = 0

    for ((token, i) <- tokens.zipWithIndex) {

      val beginOfToken = text.indexOfSlice(token, lastIndex)
      val endOfToken = (beginOfToken + token.length) - 1

      val fullTokenAnnotatorStruct = new Annotation(
        annotatorType = annotatorType,
        begin = beginOfToken,
        end = endOfToken,
        result = token,
        metadata = Map("sentence" -> sentenceIndex)
      )
      tokenTagAnnotation += fullTokenAnnotatorStruct
      lastIndex = text.indexOfSlice(token, lastIndex)
    }
    tokenTagAnnotation
  }

  private def customePOS: UserDefinedFunction = udf { (tokens: Seq[String], pos: Seq[String], text: String) =>

    val posTagAnnotation: ArrayBuffer[Annotation] = ArrayBuffer()
    def annotatorType: String = AnnotatorType.POS
    var lastIndex = 0

    for ((token, i) <- tokens.zipWithIndex) {

      val beginOfToken = text.indexOfSlice(token, lastIndex)
      val endOfToken = (beginOfToken + token.length) - 1

      val fullPOSAnnotatorStruct = new Annotation(
        annotatorType = annotatorType,
        begin = beginOfToken,
        end = endOfToken,
        result = pos(i),
        metadata = Map("word" -> token)
      )
      posTagAnnotation += fullPOSAnnotatorStruct
      lastIndex = text.indexOfSlice(token, lastIndex)
    }
    posTagAnnotation
  }

  private def wrapColumnMetadata(col: Column, annotatorType: String, outPutColName: String): Column = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)
    col.as(outPutColName, metadataBuilder.build)
  }

  private def extractData(position: Int): UserDefinedFunction = udf { docs: Seq[String] =>
    var tagsArray = ArrayBuffer[String]()
    for(e <- docs){
      val splitedArray = e.split(" ")
      tagsArray += splitedArray(position)
    }
    tagsArray
  }

  def extractMissingTokens: UserDefinedFunction= udf { (testTokens: Seq[String], predictTokens: Seq[String]) =>
    var missingTokensArray = ArrayBuffer[String]()

    for (e <- testTokens) {
      if (!predictTokens.contains(e)) {
        missingTokensArray += e
      }
    }
    missingTokensArray
  }

  def calLengthOfArray: UserDefinedFunction = udf { docs: Seq[String] =>
    docs.length
  }

  def getGoldenTokenizer(testFile: String): Dataset[_] = {
    import sparkSession.implicits._

    val textColumnName = "text"

    // Annotate the documents
    // We don't need SentenceDetector nor Tokenizer since in this example we'll use golden sentences/tokens
    val testDataSet = getTestTokensTagsDataSet(testFile)

    val documentAssembler = new DocumentAssembler()
      .setInputCol(textColumnName)
      .setOutputCol("document")
      .transform(testDataSet)

    // create Tokenizer column based on golden tokens
    documentAssembler
      .withColumn("documentText", $"document.result"(0))
      .withColumn("sentenceIndex", lit("0"))
      .withColumn("token", customeTokenizer($"testTokens", $"documentText", $"sentenceIndex"))
      .withColumn(
        "token",
        wrapColumnMetadata($"token", AnnotatorType.TOKEN, "token")
      )
      .withColumn("pos", customePOS($"testTokens", $"testPOS", $"documentText"))
      .withColumn("pos",
        wrapColumnMetadata($"pos", AnnotatorType.POS, "pos"))
      .select("id", "document", "token", "pos")

  }

  def getTestTokensTagsDataSet(testFile: String): Dataset[_] = {

    import sparkSession.implicits._

    // change the config to double newline as delimiter
    val conf = new org.apache.hadoop.mapreduce.Job().getConfiguration
    conf.set("textinputformat.record.delimiter", "\n\n")

    val usgRDD = sparkSession.sparkContext.newAPIHadoopFile(
      testFile, classOf[TextInputFormat], classOf[LongWritable], classOf[Text], conf)
      .map{ case (_, v) => v.toString }

    // remove the first line starting with DOCSTART
    val conllSentencesDF = usgRDD.map(s => s.split("\n").filter(x => !x.startsWith("-DOCSTART-")))
      .filter(x => x.length > 0)
      .toDF("sentence")

    // change the config back to its default
    conf.set("textinputformat.record.delimiter", "")

    conllSentencesDF
      .withColumn("id", monotonically_increasing_id)
      .withColumn("testTokens", extractData(TOKEN_POSITION)($"sentence"))
      .withColumn("testTags", extractData(TAG_POSITION)($"sentence"))
      .withColumn("testPOS", extractData(POS_POSITION)($"sentence"))
      .withColumn("text", concat_ws(" ", $"testTokens"))
      .withColumn("id", monotonically_increasing_id)
      .drop("sentence")
  }

}
