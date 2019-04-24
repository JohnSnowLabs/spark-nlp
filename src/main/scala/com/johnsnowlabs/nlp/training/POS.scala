package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, concat_ws, lit, split, udf}
import org.apache.spark.sql.types.MetadataBuilder

import scala.collection.mutable.ArrayBuffer

private case class TaggedToken(token: String, tag: String)
private case class TaggedDocument(sentence: String, taggedTokens: Array[TaggedToken])
private case class Annotations(text: String, document: Array[Annotation], pos: Array[Annotation])

case class POS() {

  /*
  * Add Metadata annotationType to output DataFrame
  * NOTE: This should be replaced by an existing function when it's accessible in next release
  * */

  def wrapColumnMetadata(col: Column, annotatorType: String, outPutColName: String): Column = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)
    col.as(outPutColName, metadataBuilder.build)
  }

  /*
   * This section is to help users to convert text files in token|tag style into DataFrame
   * with POS Annotation for training PerceptronApproach
   * */

  private def createDocumentAnnotation(sentence: String) = {
    Array(Annotation(
      AnnotatorType.DOCUMENT,
      0,
      sentence.length - 1,
      sentence,
      Map.empty[String, String]
    ))
  }

  private def createPosAnnotation(sentence: String, taggedTokens: Array[TaggedToken]) = {
    var lastBegin = 0
    taggedTokens.map { case TaggedToken(token, tag) =>
      val tokenBegin = sentence.indexOf(token, lastBegin)
      val a = Annotation(
        AnnotatorType.POS,
        tokenBegin,
        tokenBegin + token.length - 1,
        tag,
        Map("word" -> token)
      )
      lastBegin += token.length
      a
    }
  }

  private def lineToTaggedDocument(line: String, delimiter: String) = {
    val splitted = line.split(" ").map(_.trim)
    val tokenTags = splitted.flatMap(token => {
      val tokenTag = token.split(delimiter.head).map(_.trim)
      if (tokenTag.exists(_.isEmpty) || tokenTag.length != 2)
        // Ignore broken pairs or pairs with delimiter char
        None
      else
        Some(TaggedToken(tokenTag.head, tokenTag.last))
    })
    TaggedDocument(tokenTags.map(_.token).mkString(" "), tokenTags)
  }

  def readDataset(
                   sparkSession: SparkSession,
                   path: String,
                   delimiter: String = "|",
                   outputPosCol: String = "tags",
                   outputDocumentCol: String = "document",
                   outputTextCol: String = "text"
                 ): DataFrame = {
    import sparkSession.implicits._

    require(delimiter.length == 1, s"Delimiter must be one character long. Received $delimiter")

    val dataset = sparkSession.read.textFile(path)
      .filter(_.nonEmpty)
      .map(line => lineToTaggedDocument(line, delimiter))
      .map { case TaggedDocument(sentence, taggedTokens) =>
        Annotations(
          sentence,
          createDocumentAnnotation(sentence),
          createPosAnnotation(sentence, taggedTokens)
        )
      }

    dataset
        .withColumnRenamed(
          "text",
          outputTextCol
        )
      .withColumn(
        outputDocumentCol,
        wrapColumnMetadata(dataset("document"), AnnotatorType.DOCUMENT, outputDocumentCol)
      )
      .withColumn(
        outputPosCol,
        wrapColumnMetadata(dataset("pos"), AnnotatorType.POS, outputPosCol)
      )
      .select(outputTextCol, outputDocumentCol, outputPosCol)
  }

  // For testing purposes when there is an array of tokens and an array of labels
  def readFromDataframe(posDataframe: DataFrame, tokensCol: String = "tokens", labelsCol: String = "labels",
                        outPutDocColName: String = "text", outPutPosColName: String = "tags"): DataFrame = {
    def annotatorType: String = AnnotatorType.POS

    def annotateTokensTags: UserDefinedFunction = udf { (tokens: Seq[String], tags: Seq[String], text: String) =>
      lazy val strTokens = tokens.mkString("#")
      lazy val strPosTags = tags.mkString("#")

      require(tokens.length == tags.length, s"Cannot train from DataFrame since there" +
        s" is a row with different amount of tags and tokens:\n$strTokens\n$strPosTags")

      val tokenTagAnnotation: ArrayBuffer[Annotation] = ArrayBuffer()
      def annotatorType: String = AnnotatorType.POS
      var lastIndex = 0

      for ((e, i) <- tokens.zipWithIndex) {

        val beginOfToken = text.indexOfSlice(e, lastIndex)
        val endOfToken = (beginOfToken + e.length) - 1

        val fullPOSAnnotatorStruct = new Annotation(
          annotatorType = annotatorType,
          begin=beginOfToken,
          end=endOfToken,
          result=tags(i),
          metadata=Map("word" -> e)
        )
        tokenTagAnnotation += fullPOSAnnotatorStruct
        lastIndex = text.indexOfSlice(e, lastIndex)
      }
      tokenTagAnnotation
    }

    val tempDataFrame = posDataframe
      .withColumn(outPutDocColName,  concat_ws(" ", col(tokensCol)))
      .withColumn(outPutPosColName, annotateTokensTags(col(tokensCol), col(labelsCol), col(outPutDocColName)))
      .drop(tokensCol, labelsCol)

    tempDataFrame.withColumn(
      outPutPosColName,
      wrapColumnMetadata(tempDataFrame(outPutPosColName), annotatorType, outPutPosColName)
    )
  }

}
