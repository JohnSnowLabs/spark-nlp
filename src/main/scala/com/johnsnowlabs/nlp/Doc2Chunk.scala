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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.util.{AnnotationRowUtils, SparkNlpConfig}
import com.johnsnowlabs.util.Version
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.slf4j.LoggerFactory

/** Converts `DOCUMENT` type annotations into `CHUNK` type with the contents of a `chunkCol`.
  * Chunk text must be contained within input `DOCUMENT`. May be either `StringType` or
  * `ArrayType[StringType]` (using [[setIsArray]]). Useful for annotators that require a CHUNK
  * type input.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.{Doc2Chunk, DocumentAssembler}
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
  * val chunkAssembler = new Doc2Chunk()
  *   .setInputCols("document")
  *   .setChunkCol("target")
  *   .setOutputCol("chunk")
  *   .setIsArray(true)
  *
  * val data = Seq(
  *   ("Spark NLP is an open-source text processing library for advanced natural language processing.",
  *     Seq("Spark NLP", "text processing library", "natural language processing"))
  * ).toDF("text", "target")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, chunkAssembler)).fit(data)
  * val result = pipeline.transform(data)
  *
  * result.selectExpr("chunk.result", "chunk.annotatorType").show(false)
  * +-----------------------------------------------------------------+---------------------+
  * |result                                                           |annotatorType        |
  * +-----------------------------------------------------------------+---------------------+
  * |[Spark NLP, text processing library, natural language processing]|[chunk, chunk, chunk]|
  * +-----------------------------------------------------------------+---------------------+
  * }}}
  *
  * @see
  *   [[Chunk2Doc]] for converting `CHUNK` annotations to `DOCUMENT`
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class Doc2Chunk(override val uid: String) extends RawAnnotator[Doc2Chunk] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Output annotator types: CHUNK
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** Input annotator types: DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  private val logger = LoggerFactory.getLogger("ChunkAssembler")

  /** Column that contains string. Must be part of DOCUMENT
    *
    * @group param
    */
  val chunkCol =
    new Param[String](this, "chunkCol", "Column that contains string. Must be part of DOCUMENT")

  /** Column that has a reference of where the chunk begins
    *
    * @group param
    */
  val startCol =
    new Param[String](this, "startCol", "Column that has a reference of where the chunk begins")

  /** Whether start col is by whitespace tokens (Default: `false`)
    *
    * @group param
    */
  val startColByTokenIndex = new BooleanParam(
    this,
    "startColByTokenIndex",
    "Whether start col is by whitespace tokens (Default: `false`)")

  /** Whether the chunkCol is an array of strings (Default: `false`)
    *
    * @group param
    */
  val isArray = new BooleanParam(
    this,
    "isArray",
    "Whether the chunkCol is an array of strings (Default: `false")

  /** Whether to fail the job if a chunk is not found within document, return empty otherwise
    * (Default: `false`)
    *
    * @group param
    */
  val failOnMissing = new BooleanParam(
    this,
    "failOnMissing",
    "Whether to fail the job if a chunk is not found within document, return empty otherwise (Default: `false`)")

  /** Whether to lower case for matching case (Default: `true`)
    *
    * @group param
    */
  val lowerCase =
    new BooleanParam(this, "lowerCase", "Whether to lower case for matching case (Default: `true")

  setDefault(
    startColByTokenIndex -> false,
    isArray -> false,
    failOnMissing -> false,
    lowerCase -> true)

  /** Column that contains string. Must be part of DOCUMENT
    *
    * @group setParam
    */
  def setChunkCol(value: String): this.type = set(chunkCol, value)

  /** Column that contains string. Must be part of DOCUMENT
    *
    * @group getParam
    */
  def getChunkCol: String = $(chunkCol)

  /** Column that has a reference of where the chunk begins
    *
    * @group setParam
    */
  def setStartCol(value: String): this.type = set(startCol, value)

  /** Column that has a reference of where the chunk begins
    *
    * @group getParam
    */
  def getStartCol: String = $(startCol)

  /** Whether start col is by whitespace tokens (Default: `false`)
    *
    * @group setParam
    */
  def setStartColByTokenIndex(value: Boolean): this.type = set(startColByTokenIndex, value)

  /** Whether start col is by whitespace tokens (Default: `false`)
    *
    * @group getParam
    */
  def getStartColByTokenIndex: Boolean = $(startColByTokenIndex)

  /** Whether the chunkCol is an array of strings (Default: `false`)
    *
    * @group setParam
    */
  def setIsArray(value: Boolean): this.type = set(isArray, value)

  /** Whether the chunkCol is an array of strings (Default: `false`)
    *
    * @group getParam
    */
  def getIsArray: Boolean = $(isArray)

  /** Whether to fail the job if a chunk is not found within document, return empty otherwise
    * (Default: `false`)
    *
    * @group setParam
    */
  def setFailOnMissing(value: Boolean): this.type = set(failOnMissing, value)

  /** Whether to fail the job if a chunk is not found within document, return empty otherwise
    * (Default: `false`)
    *
    * @group getParam
    */
  def getFailOnMissing: Boolean = $(failOnMissing)

  /** Whether to lower case for matching case (Default: `true`)
    *
    * @group setParam
    */
  def setLowerCase(value: Boolean): this.type = set(lowerCase, value)

  /** Whether to lower case for matching case (Default: `true`)
    *
    * @group getParam
    */
  def getLowerCase: Boolean = $(lowerCase)

  def this() = this(Identifiable.randomUID("DOC2CHUNK"))

  override protected def extraValidate(structType: StructType): Boolean = {
    if (get(chunkCol).isEmpty)
      true
    else if ($(isArray))
      structType.fields
        .find(_.name == $(chunkCol))
        .exists(_.dataType == ArrayType(StringType, containsNull = true))
    else
      structType.fields.find(_.name == $(chunkCol)).exists(_.dataType == StringType)
  }

  override protected def extraValidateMsg: AnnotatorType =
    if ($(isArray)) s"${$(chunkCol)} must be ArrayType(StringType)"
    else s"${$(chunkCol)} must be StringType"

  private def buildFromChunk(
      annotation: Annotation,
      chunk: String,
      startIndex: Int,
      chunkIdx: Int) = {

    /** This will break if there are two identical chunks */
    val beginning = get(lowerCase) match {
      case Some(true) => annotation.result.toLowerCase.indexOf(chunk, startIndex)
      case _ => annotation.result.indexOf(chunk, startIndex)
    }
    val ending = beginning + chunk.length - 1
    if (chunk.trim.isEmpty || beginning == -1) {
      val message =
        s"Cannot proceed to assemble CHUNK, because could not find: `$chunk` within: `${annotation.result}`"
      if ($(failOnMissing))
        throw new Exception(message)
      else
        logger.warn(message)
      None
    } else {
      Some(
        Annotation(
          outputAnnotatorType,
          beginning,
          ending,
          chunk,
          annotation.metadata ++ Map("chunk" -> chunkIdx.toString)))
    }
  }

  def tokenIndexToCharIndex(text: String, tokenIndex: Int): Int = {
    var i = 0
    text
      .split(" ")
      .map(token => {
        val o = (token, i)
        i += token.length + 1
        o
      })
      .apply(tokenIndex)
      ._2
  }

  private def convertDocumentToChunkAnnotations(document: Seq[Row]): Seq[Annotation] = {
    val annotations = document.map(Annotation(_))
    annotations.map { annotation =>
      Annotation(
        AnnotatorType.CHUNK,
        annotation.begin,
        annotation.end,
        annotation.result,
        annotation.metadata ++ Map("chunk" -> "0"))
    }
  }

  private def assembleChunksAnnotations(
      annotationProperties: Seq[Row],
      chunks: Seq[String]): Seq[Annotation] = {
    val annotations = annotationProperties.map(Annotation(_))
    annotations.flatMap(annotation => {
      chunks.zipWithIndex.flatMap { case (chunk, idx) =>
        buildFromChunk(annotation, chunk, 0, idx)
      }
    })
  }

  private def assembleChunkAnnotations(
      annotationProperties: Seq[Row],
      chunk: String): Seq[Annotation] = {
    val annotations = annotationProperties.map(Annotation(_))
    annotations.flatMap(annotation => {
      buildFromChunk(annotation, chunk, 0, 0)
    })
  }

  private def assembleChunkWithStartAnnotations(
      annotationProperties: Seq[Row],
      chunk: String,
      start: Int): Seq[Annotation] = {
    val annotations = annotationProperties.map(Annotation(_))
    annotations.flatMap(annotation => {
      if ($(startColByTokenIndex))
        buildFromChunk(annotation, chunk, tokenIndexToCharIndex(annotation.result, start), 0)
      else
        buildFromChunk(annotation, chunk, start, 0)
    })
  }

  private def convertDocumentToChunk = udf { document: Seq[Row] =>
    convertDocumentToChunkAnnotations(document)
  }

  private def assembleChunks = udf { (annotationProperties: Seq[Row], chunks: Seq[String]) =>
    assembleChunksAnnotations(annotationProperties, chunks)
  }

  private def assembleChunk = udf { (annotationProperties: Seq[Row], chunk: String) =>
    assembleChunkAnnotations(annotationProperties, chunk)
  }

  private def assembleChunkWithStart = udf {
    (annotationProperties: Seq[Row], chunk: String, start: Int) =>
      assembleChunkWithStartAnnotations(annotationProperties, chunk, start)
  }

  private def isSpark4OrNewer(dataset: Dataset[_]): Boolean =
    Version.parse(dataset.sparkSession.version).toFloat >= 4.0f

  private def transformWithRows(dataset: Dataset[_]): DataFrame = {
    val inputDataFrame = dataset.toDF()
    val outputSchema = inputDataFrame.schema.add($(outputCol), Annotation.arrayType)
    val documentIndex = inputDataFrame.schema.fieldIndex(getInputCols.head)
    val chunkIndex = get(chunkCol).map(inputDataFrame.schema.fieldIndex)
    val startIndex = get(startCol).map(inputDataFrame.schema.fieldIndex)

    implicit val encoder: ExpressionEncoder[Row] =
      SparkNlpConfig.getEncoder(inputDataFrame, outputSchema)

    val mappedDataFrame = inputDataFrame.mapPartitions { rows =>
      rows.map { row =>
        val annotationProperties =
          AnnotationRowUtils.extractAnnotationRows(row, documentIndex).toVector

        val outputAnnotations =
          if (get(chunkCol).isEmpty) {
            convertDocumentToChunkAnnotations(annotationProperties)
          } else if ($(isArray)) {
            assembleChunksAnnotations(annotationProperties, row.getSeq[String](chunkIndex.get))
          } else if (startIndex.isDefined) {
            assembleChunkWithStartAnnotations(
              annotationProperties,
              row.getString(chunkIndex.get),
              row.getInt(startIndex.get))
          } else {
            assembleChunkAnnotations(annotationProperties, row.getString(chunkIndex.get))
          }

        Row.fromSeq(row.toSeq :+ AnnotationRowUtils.annotationsToRows(outputAnnotations).toVector)
      }
    }

    val withInputMetadata = inputDataFrame.schema.fields
      .filter(field => mappedDataFrame.columns.contains(field.name))
      .foldLeft(mappedDataFrame)((dataFrame, field) => {
        dataFrame.withColumn(field.name, dataFrame.col(field.name).as(field.name, field.metadata))
      })

    withInputMetadata.withColumn($(outputCol), wrapColumnMetadata(col($(outputCol))))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (isSpark4OrNewer(dataset)) {
      transformWithRows(dataset)
    } else if (get(chunkCol).isEmpty)
      dataset.withColumn(
        $(outputCol),
        wrapColumnMetadata(convertDocumentToChunk(col(getInputCols.head))))
    else if ($(isArray))
      dataset.withColumn(
        $(outputCol),
        wrapColumnMetadata(assembleChunks(col(getInputCols.head), col($(chunkCol)))))
    else if (get(startCol).isDefined)
      dataset.withColumn(
        $(outputCol),
        wrapColumnMetadata(
          assembleChunkWithStart(col($(inputCols).head), col($(chunkCol)), col($(startCol)))))
    else
      dataset.withColumn(
        $(outputCol),
        wrapColumnMetadata(assembleChunk(col(getInputCols.head), col($(chunkCol)))))
  }

}

/** This is the companion object of [[Doc2Chunk]]. Please refer to that class for the
  * documentation.
  */
object Doc2Chunk extends DefaultParamsReadable[Doc2Chunk]
