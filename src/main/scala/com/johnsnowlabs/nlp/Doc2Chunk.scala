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
  val isArray = new BooleanParam(this, "isArray", "whether the chunkCol is an array of strings")

  setDefault(isArray -> false)

  def setChunkCol(value: String): this.type = set(chunkCol, value)
  def setIsArray(value: Boolean): this.type = set(isArray, value)

  def getChunkCol: String = $(chunkCol)
  def getIsArray: Boolean = $(isArray)

  def this() = this(Identifiable.randomUID("CHUNK_ASSEMBLER"))

  override protected def extraValidate(structType: StructType): Boolean = {
    if ($(isArray))
      structType.fields.find(_.name == $(chunkCol)).exists(_.dataType == ArrayType(StringType, containsNull=true))
    else
      structType.fields.find(_.name == $(chunkCol)).exists(_.dataType == StringType)
  }

  override protected def extraValidateMsg: AnnotatorType =
    if ($(isArray)) s"${$(chunkCol)} must be ArrayType(StringType)"
    else s"${$(chunkCol)} must be StringType"

  private def buildFromChunk(annotation: Annotation, chunk: String) = {
    /** This will break if there are two identical chunks */
    val beginning = annotation.result.indexOf(chunk)
    val ending = beginning + chunk.length - 1
    if (chunk.trim.isEmpty || beginning == -1) {
      logger.warn(s"Cannot proceed to assemble CHUNK, because could not find: `$chunk` within: `${annotation.result}`")
      None
    } else {
      Some(Annotation(
        outputAnnotatorType,
        beginning,
        ending,
        chunk,
        annotation.metadata
      ))
    }
  }

  private def assembleChunks = udf {
    (annotationProperties: Seq[Row], chunks: Seq[String]) =>
      val annotations = annotationProperties.map(Annotation(_))
      annotations.flatMap(annotation => {
        chunks.flatMap(chunk => buildFromChunk(annotation, chunk))
      })
  }

  private def assembleChunk = udf {
    (annotationProperties: Seq[Row], chunk: String) =>
      val annotations = annotationProperties.map(Annotation(_))
      annotations.flatMap(annotation => {
        buildFromChunk(annotation, chunk)
      })
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    if ($(isArray))
      dataset.withColumn($(outputCol), wrapColumnMetadata(assembleChunks(col(getInputCols.head), col($(chunkCol)))))
    else
      dataset.withColumn($(outputCol), wrapColumnMetadata(assembleChunk(col(getInputCols.head), col($(chunkCol)))))
  }

}
object Doc2Chunk extends DefaultParamsReadable[Doc2Chunk]
