/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.nlp.util.io.OutputHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, concat_ws, udf}
import org.apache.spark.sql.types.MetadataBuilder
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

import scala.collection.mutable.ArrayBuffer

private case class TaggedToken(token: String, tag: String)
private case class TaggedDocument(sentence: String, taggedTokens: Array[TaggedToken])
private case class Annotations(text: String, document: Array[Annotation], pos: Array[Annotation])

/** Helper class for creating DataFrames for training a part-of-speech tagger.
  *
  * The dataset needs to consist of sentences on each line, where each word is delimited with its
  * respective tag:
  *
  * {{{
  * Pierre|NNP Vinken|NNP ,|, 61|CD years|NNS old|JJ ,|, will|MD join|VB the|DT board|NN as|IN a|DT nonexecutive|JJ director|NN Nov.|NNP 29|CD .|.
  * }}}
  *
  * The sentence can then be parsed with [[readDataset]] into a column with annotations of type
  * `POS`.
  *
  * ==Example==
  * In this example, the file `test-training.txt` has the content of the sentence above.
  * {{{
  * import com.johnsnowlabs.nlp.training.POS
  *
  * val pos = POS()
  * val path = "src/test/resources/anc-pos-corpus-small/test-training.txt"
  * val posDf = pos.readDataset(spark, path, "|", "tags")
  *
  * posDf.selectExpr("explode(tags) as tags").show(false)
  * +---------------------------------------------+
  * |tags                                         |
  * +---------------------------------------------+
  * |[pos, 0, 5, NNP, [word -> Pierre], []]       |
  * |[pos, 7, 12, NNP, [word -> Vinken], []]      |
  * |[pos, 14, 14, ,, [word -> ,], []]            |
  * |[pos, 16, 17, CD, [word -> 61], []]          |
  * |[pos, 19, 23, NNS, [word -> years], []]      |
  * |[pos, 25, 27, JJ, [word -> old], []]         |
  * |[pos, 29, 29, ,, [word -> ,], []]            |
  * |[pos, 31, 34, MD, [word -> will], []]        |
  * |[pos, 36, 39, VB, [word -> join], []]        |
  * |[pos, 41, 43, DT, [word -> the], []]         |
  * |[pos, 45, 49, NN, [word -> board], []]       |
  * |[pos, 51, 52, IN, [word -> as], []]          |
  * |[pos, 47, 47, DT, [word -> a], []]           |
  * |[pos, 56, 67, JJ, [word -> nonexecutive], []]|
  * |[pos, 69, 76, NN, [word -> director], []]    |
  * |[pos, 78, 81, NNP, [word -> Nov.], []]       |
  * |[pos, 83, 84, CD, [word -> 29], []]          |
  * |[pos, 81, 81, ., [word -> .], []]            |
  * +---------------------------------------------+
  * }}}
  */
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
    Array(
      Annotation(
        AnnotatorType.DOCUMENT,
        0,
        sentence.length - 1,
        sentence,
        Map.empty[String, String]))
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
        Map("word" -> token))
      lastBegin += token.length
      a
    }
  }

  private def lineToTaggedDocument(line: String, delimiter: String) = {

    /*
    TODO: improve the performance of regex group
    val splitted = line.replaceAll(s"(?:${delimiter.head}\\w+)+(\\s)", "$0##$1").split("##").map(_.trim)
     */
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

  /** Reads the provided dataset file with given parameters and returns a DataFrame ready to for
    * training a part-of-speech tagger.
    *
    * @param sparkSession
    *   Current Spark sessions
    * @param path
    *   Path to the resource
    * @param delimiter
    *   Delimiter used to separate word from their tag in the text
    * @param outputPosCol
    *   Name for the output column of the part-of-tags
    * @param outputDocumentCol
    *   Name for the [[com.johnsnowlabs.nlp.base.DocumentAssembler DocumentAssembler]] column
    * @param outputTextCol
    *   Name for the column of the raw text
    * @return
    *   DataFrame of parsed text
    */
  def readDataset(
      sparkSession: SparkSession,
      path: String,
      delimiter: String = "|",
      outputPosCol: String = "tags",
      outputDocumentCol: String = "document",
      outputTextCol: String = "text"): DataFrame = {
    import sparkSession.implicits._

    require(delimiter.length == 1, s"Delimiter must be one character long. Received $delimiter")

    val dataset = sparkSession.read
      .textFile(OutputHelper.parsePath(path))
      .filter(_.nonEmpty)
      .map(line => lineToTaggedDocument(line, delimiter))
      .map { case TaggedDocument(sentence, taggedTokens) =>
        Annotations(
          sentence,
          createDocumentAnnotation(sentence),
          createPosAnnotation(sentence, taggedTokens))
      }

    dataset
      .withColumnRenamed("text", outputTextCol)
      .withColumn(
        outputDocumentCol,
        wrapColumnMetadata(dataset("document"), AnnotatorType.DOCUMENT, outputDocumentCol))
      .withColumn(
        outputPosCol,
        wrapColumnMetadata(dataset("pos"), AnnotatorType.POS, outputPosCol))
      .select(outputTextCol, outputDocumentCol, outputPosCol)
  }

  // For testing purposes when there is an array of tokens and an array of labels
  def readFromDataframe(
      posDataframe: DataFrame,
      tokensCol: String = "tokens",
      labelsCol: String = "labels",
      outPutDocColName: String = "text",
      outPutPosColName: String = "tags"): DataFrame = {
    def annotatorType: String = AnnotatorType.POS

    def annotateTokensTags: UserDefinedFunction = udf {
      (tokens: Seq[String], tags: Seq[String], text: String) =>
        lazy val strTokens = tokens.mkString("#")
        lazy val strPosTags = tags.mkString("#")

        require(
          tokens.length == tags.length,
          s"Cannot train from DataFrame since there" +
            s" is a row with different amount of tags and tokens:\n$strTokens\n$strPosTags")

        val tokenTagAnnotation: ArrayBuffer[Annotation] = ArrayBuffer()
        def annotatorType: String = AnnotatorType.POS
        var lastIndex = 0

        for ((e, i) <- tokens.zipWithIndex) {

          val beginOfToken = text.indexOfSlice(e, lastIndex)
          val endOfToken = (beginOfToken + e.length) - 1

          val fullPOSAnnotatorStruct = new Annotation(
            annotatorType = annotatorType,
            begin = beginOfToken,
            end = endOfToken,
            result = tags(i),
            metadata = Map("word" -> e))
          tokenTagAnnotation += fullPOSAnnotatorStruct
          lastIndex = text.indexOfSlice(e, lastIndex)
        }
        tokenTagAnnotation
    }

    val tempDataFrame = posDataframe
      .withColumn(outPutDocColName, concat_ws(" ", col(tokensCol)))
      .withColumn(
        outPutPosColName,
        annotateTokensTags(col(tokensCol), col(labelsCol), col(outPutDocColName)))
      .drop(tokensCol, labelsCol)

    tempDataFrame.withColumn(
      outPutPosColName,
      wrapColumnMetadata(tempDataFrame(outPutPosColName), annotatorType, outPutPosColName))
  }

}
