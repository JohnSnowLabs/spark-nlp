package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, concat_ws, lit, split, udf}
import org.apache.spark.sql.types.MetadataBuilder

import scala.collection.mutable.ArrayBuffer

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

  private def annotateTokensTags: UserDefinedFunction = udf { (tokens: Seq[String], tags: Seq[String], text: String) =>
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

  private def extractTokensAndTags: UserDefinedFunction = udf { (tokensTags: Seq[String], delimiter: String, condition: String) =>

    val tempArray: ArrayBuffer[String] = ArrayBuffer()

    for (e <- tokensTags.zipWithIndex) {
      val splittedTokenTag: Array[String] = e._1.split(delimiter.mkString)
      if(splittedTokenTag.length > 1){
        condition.mkString match {
          case "token" =>
            tempArray += splittedTokenTag(0)

          case "tag" =>
            tempArray += splittedTokenTag(1)
        }
      }
    }
    tempArray
  }

  def readDataset(sparkSession: SparkSession, path: String, delimiter: String = "|", outputPosCol: String = "tags", outputDocumentCol: String = "text"): DataFrame = {
    import sparkSession.implicits._
    def annotatorType: String = AnnotatorType.POS

    val tempDataFrame = sparkSession.read.text(path).toDF
      .filter(row => !(row.mkString("").isEmpty && row.length>0))
      .withColumn("token_tags", split($"value", " "))
      .select("token_tags")
      .withColumn("tokens", extractTokensAndTags($"token_tags", lit(delimiter), lit("token")))
      .withColumn("tags", extractTokensAndTags($"token_tags", lit(delimiter), lit("tag")))
      .withColumn(outputDocumentCol,  concat_ws(" ", $"tokens"))
      .withColumn(outputPosCol, annotateTokensTags($"tokens", $"tags", col(outputDocumentCol)))
      .drop("tokens", "token_tags")

    tempDataFrame.withColumn(
      outputPosCol,
      wrapColumnMetadata(tempDataFrame(outputPosCol), annotatorType, outputPosCol)
    )
  }

  // For testing purposes when there is an array of tokens and an array of labels
  def readDatframe(posDataframe: DataFrame, tokensCol: String = "tokens", labelsCol: String = "labels",
                   outPutDocColName: String = "text", outPutPosColName: String = "tags"): DataFrame = {
    def annotatorType: String = AnnotatorType.POS

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
