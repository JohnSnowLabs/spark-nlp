package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, MetadataBuilder, StringType, StructType}
import org.slf4j.LoggerFactory

/**
  * Created by saif on 06/07/17.
  */

class Doc2Chunk(override val uid: String) extends RawAnnotator[Doc2Chunk]{

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val outputAnnotatorType: AnnotatorType = CHUNK

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  private val logger = LoggerFactory.getLogger("ChunkAssembler")

  val chunkCol = new Param[String](this, "chunkCol", "column that contains string. Must be part of DOCUMENT")
  val startCol = new Param[String](this, "startCol", "column that has a reference of where chunk begins")
  val startColByTokenIndex = new BooleanParam(this, "startColByTokenIndex", "whether start col is by whitespace tokens")
  val isArray = new BooleanParam(this, "isArray", "whether the chunkCol is an array of strings")
  val failOnMissing = new BooleanParam(this, "failOnMissing", "whether to fail the job if a chunk is not found within document. return empty otherwise")
  val lowerCase = new BooleanParam(this, "lowerCase", "whether to lower case for matching case")

  setDefault(
    startColByTokenIndex -> false,
    isArray -> false,
    failOnMissing -> false,
    lowerCase -> true
  )

  def setChunkCol(value: String): this.type = set(chunkCol, value)
  def setIsArray(value: Boolean): this.type = set(isArray, value)

  def getChunkCol: String = $(chunkCol)
  def getIsArray: Boolean = $(isArray)

  def setStartCol(value: String): this.type = set(startCol, value)
  def getStartCol: String = $(startCol)

  def setStartColByTokenIndex(value: Boolean): this.type = set(startColByTokenIndex, value)
  def getStartColByTokenIndex: Boolean = $(startColByTokenIndex)

  def setFailOnMissing(value: Boolean): this.type = set(failOnMissing, value)
  def getFailOnMissing: Boolean = $(failOnMissing)

  def setLowerCase(value: Boolean): this.type = set(lowerCase, value)
  def getLowerCase: Boolean = $(lowerCase)

  def this() = this(Identifiable.randomUID("DOC2CHUNK"))

  override protected def extraValidate(structType: StructType): Boolean = {
    if (get(chunkCol).isEmpty)
      true
    else if ($(isArray))
      structType.fields.find(_.name == $(chunkCol)).exists(_.dataType == ArrayType(StringType, containsNull=true))
    else
      structType.fields.find(_.name == $(chunkCol)).exists(_.dataType == StringType)
  }

  override protected def extraValidateMsg: AnnotatorType =
    if ($(isArray)) s"${$(chunkCol)} must be ArrayType(StringType)"
    else s"${$(chunkCol)} must be StringType"

  private def buildFromChunk(annotation: Annotation, chunk: String, startIndex: Int, chunkIdx: Int) = {
    /** This will break if there are two identical chunks */
    val beginning = get(lowerCase) match {
      case Some(true) => annotation.result.toLowerCase.indexOf(chunk, startIndex)
      case _ => annotation.result.indexOf(chunk, startIndex)
    }
    val ending = beginning + chunk.length - 1
    if (chunk.trim.isEmpty || beginning == -1) {
      val message = s"Cannot proceed to assemble CHUNK, because could not find: `$chunk` within: `${annotation.result}`"
      if ($(failOnMissing))
        throw new Exception(message)
      else
        logger.warn(message)
      None
    } else {
      Some(Annotation(
        outputAnnotatorType,
        beginning,
        ending,
        chunk,
        annotation.metadata ++ Map("chunk" -> chunkIdx.toString)
      ))
    }
  }

  def tokenIndexToCharIndex(text: String, tokenIndex: Int): Int = {
    var i = 0
    text.split(" ").map(token => {
      val o = (token, i)
      i += token.length + 1
      o
    }).apply(tokenIndex)._2
  }

  private def convertDocumentToChunk = udf {
    document: Seq[Row] =>
      val annotations = document.map(Annotation(_))
      annotations.map{annotation =>
        Annotation(AnnotatorType.CHUNK, annotation.begin, annotation.end, annotation.result, annotation.metadata)
      }
  }

  private def assembleChunks = udf {
    (annotationProperties: Seq[Row], chunks: Seq[String]) =>
      val annotations = annotationProperties.map(Annotation(_))
      annotations.flatMap(annotation => {
        chunks.zipWithIndex.flatMap{case (chunk, idx) => buildFromChunk(annotation, chunk, 0, idx)}
      })
  }

  private def assembleChunk = udf {
    (annotationProperties: Seq[Row], chunk: String) =>
      val annotations = annotationProperties.map(Annotation(_))
      annotations.flatMap(annotation => {
        buildFromChunk(annotation, chunk, 0, 0)
      })
  }

  private def assembleChunkWithStart = udf {
    (annotationProperties: Seq[Row], chunk: String, start: Int) =>
      val annotations = annotationProperties.map(Annotation(_))
      annotations.flatMap(annotation => {
        if ($(startColByTokenIndex))
          buildFromChunk(annotation, chunk, tokenIndexToCharIndex(annotation.result, start), 0)
        else
          buildFromChunk(annotation, chunk, start, 0)
      })
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (get(chunkCol).isEmpty)
      dataset.withColumn($(outputCol), wrapColumnMetadata(convertDocumentToChunk(col(getInputCols.head))))
    else if ($(isArray))
      dataset.withColumn($(outputCol), wrapColumnMetadata(assembleChunks(col(getInputCols.head), col($(chunkCol)))))
    else if (get(startCol).isDefined)
      dataset.withColumn($(outputCol), wrapColumnMetadata(assembleChunkWithStart(
        col($(inputCols).head),
        col($(chunkCol)),
        col($(startCol))
      )))
    else
      dataset.withColumn($(outputCol), wrapColumnMetadata(assembleChunk(col(getInputCols.head), col($(chunkCol)))))
  }

}
object Doc2Chunk extends DefaultParamsReadable[Doc2Chunk]
