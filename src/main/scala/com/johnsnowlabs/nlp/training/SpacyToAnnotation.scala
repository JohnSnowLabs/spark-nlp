/*
 * Copyright 2017-2023 John Snow Labs
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

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

class SpacyToAnnotation() {

  /** Helper class to load a list of tokens/sentences as JSON to Annotation.
    *
    * The JSON will be in this format:
    *
    * [ { "tokens": ["Hello", "world", "!", "How", "are", "you", "today", "?", "I", "'m", "fine",
    * "thanks", "."], "token_spaces": [true, false, true, true, true, true, false, true, false,
    * true, true, false, false], "sentence_ends": [2, 7, 12] } ]
    *
    * sentence_ends is optional
    *
    * This format can be exported from spaCy check this notebook for details:
    *
    * ==Example==
    * {{{
    * val nlpReader = new SpacyToAnnotation()
    * val result = nlpReader.readJsonFile(spark, "src/test/resources/spacy-to-annotation/multi_doc_tokens.json")
    * result.show(false)
    *
    * +-------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |document                                                                             |sentence                                                                                                                                                                      |token                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
    * +-------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |[{document, 0, 55, John went to the store last night. He bought some bread., {}, []}]|[{document, 0, 33, John went to the store last night., {sentence -> 0}, []}, {document, 35, 55, He bought some bread., {sentence -> 1}, []}]                                  |[{token, 0, 3, John, {sentence -> 0}, []}, {token, 5, 8, went, {sentence -> 0}, []}, {token, 10, 11, to, {sentence -> 0}, []}, {token, 13, 15, the, {sentence -> 0}, []}, {token, 17, 21, store, {sentence -> 0}, []}, {token, 23, 26, last, {sentence -> 0}, []}, {token, 28, 32, night, {sentence -> 0}, []}, {token, 33, 33, ., {sentence -> 0}, []}, {token, 35, 36, He, {sentence -> 1}, []}, {token, 38, 43, bought, {sentence -> 1}, []}, {token, 45, 48, some, {sentence -> 1}, []}, {token, 50, 54, bread, {sentence -> 1}, []}, {token, 55, 55, ., {sentence -> 1}, []}]|
    * |[{document, 0, 47, Hello world! How are you today? I'm fine thanks., {}, []}]        |[{document, 0, 11, Hello world!, {sentence -> 0}, []}, {document, 13, 30, How are you today?, {sentence -> 1}, []}, {document, 32, 47, I'm fine thanks., {sentence -> 2}, []}]|[{token, 0, 4, Hello, {sentence -> 0}, []}, {token, 6, 10, world, {sentence -> 0}, []}, {token, 11, 11, !, {sentence -> 0}, []}, {token, 13, 15, How, {sentence -> 1}, []}, {token, 17, 19, are, {sentence -> 1}, []}, {token, 21, 23, you, {sentence -> 1}, []}, {token, 25, 29, today, {sentence -> 1}, []}, {token, 30, 30, ?, {sentence -> 1}, []}, {token, 32, 32, I, {sentence -> 2}, []}, {token, 33, 34, 'm, {sentence -> 2}, []}, {token, 36, 39, fine, {sentence -> 2}, []}, {token, 41, 46, thanks, {sentence -> 2}, []}, {token, 47, 47, ., {sentence -> 2}, []}]     |
    * +-------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    *
    *
    * result.printSchema
    * root
    *  |-- document: array (nullable = true)
    *  |    |-- element: struct (containsNull = true)
    *  |    |    |-- annotatorType: string (nullable = true)
    *  |    |    |-- begin: integer (nullable = false)
    *  |    |    |-- end: integer (nullable = false)
    *  |    |    |-- result: string (nullable = true)
    *  |    |    |-- metadata: map (nullable = true)
    *  |    |    |    |-- key: string
    *  |    |    |    |-- value: string (valueContainsNull = true)
    *  |    |    |-- embeddings: array (nullable = true)
    *  |    |    |    |-- element: float (containsNull = false)
    *  |-- sentence: array (nullable = true)
    *  |    |-- element: struct (containsNull = true)
    *  |    |    |-- annotatorType: string (nullable = true)
    *  |    |    |-- begin: integer (nullable = false)
    *  |    |    |-- end: integer (nullable = false)
    *  |    |    |-- result: string (nullable = true)
    *  |    |    |-- metadata: map (nullable = true)
    *  |    |    |    |-- key: string
    *  |    |    |    |-- value: string (valueContainsNull = true)
    *  |    |    |-- embeddings: array (nullable = true)
    *  |    |    |    |-- element: float (containsNull = false)
    *  |-- token: array (nullable = true)
    *  |    |-- element: struct (containsNull = true)
    *  |    |    |-- annotatorType: string (nullable = true)
    *  |    |    |-- begin: integer (nullable = false)
    *  |    |    |-- end: integer (nullable = false)
    *  |    |    |-- result: string (nullable = true)
    *  |    |    |-- metadata: map (nullable = true)
    *  |    |    |    |-- key: string
    *  |    |    |    |-- value: string (valueContainsNull = true)
    *  |    |    |-- embeddings: array (nullable = true)
    *  |    |    |    |-- element: float (containsNull = false)
    * }}}
    */

  def readJsonFileJava(
      spark: SparkSession,
      jsonFilePath: String,
      params: java.util.Map[String, String]): Dataset[_] = {
    readJsonFile(spark, jsonFilePath, params.asScala.toMap)
  }

  def readJsonFile(
      spark: SparkSession,
      jsonFilePath: String,
      params: Map[String, String] = Map()): Dataset[_] = {

    val outputAnnotatorType = params.getOrElse("outputAnnotatorType", TOKEN)

    val availableAnnotatorTypes = Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)
    if (!availableAnnotatorTypes.contains(outputAnnotatorType.toLowerCase)) {
      throw new IllegalArgumentException(
        s"Cannot convert Annotator Type: $outputAnnotatorType. Not yet supported")
    }

    if (!ResourceHelper.validFile(jsonFilePath)) {
      throw new IllegalArgumentException(s"Invalid jsonFilePath: $jsonFilePath. Please verify")
    }

    val inputDataset = spark.read.option("multiline", "true").json(jsonFilePath)

    validateSchema(inputDataset)

    if (inputDataset.schema.fieldNames.toSet.contains("sentence_ends")) {
      val annotationDataset = inputDataset.withColumn(
        "annotations",
        buildTokenAnnotationsWithSentences(
          col("tokens"),
          col("token_spaces"),
          col("sentence_ends")))

      annotationDataset
        .select("annotations.*")
        .withColumnRenamed("_1", "document")
        .withColumnRenamed("_2", "sentence")
        .withColumnRenamed("_3", "token")
    } else {
      val annotationDataset = inputDataset.withColumn(
        "annotations",
        buildTokenAnnotationsWithoutSentences(col("tokens"), col("token_spaces")))

      annotationDataset
        .select("annotations.*")
        .withColumnRenamed("_1", "document")
        .withColumnRenamed("_2", "token")
    }

  }

  private def validateSchema(dataset: Dataset[_]): Unit = {
    val expectedSchema = StructType(
      Array(
        StructField("tokens", ArrayType(StringType, false), false),
        StructField("token_spaces", ArrayType(BooleanType, false), false)))

    val expectedFieldNames = expectedSchema.fieldNames.toSet

    val actualFieldNames =
      dataset.schema.fieldNames.toSet.filter(fieldName => fieldName != "sentence_ends")
    if (actualFieldNames != expectedFieldNames) {
      throw new IllegalArgumentException(
        s"Schema validation failed. Expected field names: ${expectedFieldNames.mkString(
            ", ")}, actual field names: ${actualFieldNames.mkString(", ")}")
    }
  }

  private def buildTokenAnnotationsWithSentences: UserDefinedFunction =
    udf((tokens: Seq[String], tokenSpaces: Seq[Boolean], sentenceEnds: Seq[Long]) => {
      val stringBuilder = new StringBuilder
      val sentences = ArrayBuffer[String]()
      val sentencesAnnotations = ArrayBuffer[Annotation]()
      val tokenAnnotations = ArrayBuffer[Annotation]()
      var beginToken = 0
      var beginSentence = 0
      var sentenceIndex = 0

      tokens.zip(tokenSpaces).zipWithIndex.foreach { case ((token, tokenSpace), index) =>
        stringBuilder.append(token)
        val endToken = beginToken + token.length - 1
        tokenAnnotations += Annotation(
          AnnotatorType.TOKEN,
          beginToken,
          endToken,
          token,
          Map("sentence" -> sentenceIndex.toString))
        if (tokenSpace) {
          beginToken = beginToken + token.length + 1
          stringBuilder.append(" ")
        } else {
          beginToken = beginToken + token.length
        }
        if (sentenceEnds.contains(index)) {
          sentences += stringBuilder.toString
          val endSentence = beginSentence + sentences.last.trim.length - 1
          sentencesAnnotations += Annotation(
            AnnotatorType.DOCUMENT,
            beginSentence,
            endSentence,
            sentences.last.trim,
            Map("sentence" -> sentenceIndex.toString))
          beginSentence = beginSentence + sentences.last.trim.length + 1
          sentenceIndex = sentenceIndex + 1
          stringBuilder.clear()
        }
      }

      val result = sentencesAnnotations.map(annotation => annotation.result).mkString(" ")
      val documentAnnotation =
        Array(Annotation(AnnotatorType.DOCUMENT, 0, result.length - 1, result, Map()))

      (documentAnnotation, sentencesAnnotations.toArray, tokenAnnotations.toArray)
    })

  private def buildTokenAnnotationsWithoutSentences: UserDefinedFunction =
    udf((tokens: Seq[String], tokenSpaces: Seq[Boolean]) => {
      val stringBuilder = new StringBuilder
      val tokenAnnotations = ArrayBuffer[Annotation]()
      var beginToken = 0

      tokens.zip(tokenSpaces).foreach { case (token, tokenSpace) =>
        stringBuilder.append(token)
        val endToken = beginToken + token.length - 1
        tokenAnnotations += Annotation(
          AnnotatorType.TOKEN,
          beginToken,
          endToken,
          token,
          Map("sentence" -> "0"))
        if (tokenSpace) {
          beginToken = beginToken + token.length + 1
          stringBuilder.append(" ")
        } else {
          beginToken = beginToken + token.length
        }
      }

      val result = stringBuilder.toString()
      val documentAnnotation =
        Array(Annotation(AnnotatorType.DOCUMENT, 0, result.length - 1, result, Map()))

      (documentAnnotation, tokenAnnotations.toArray)
    })

}
