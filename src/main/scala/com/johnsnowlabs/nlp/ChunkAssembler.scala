package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.MetadataBuilder
import org.slf4j.LoggerFactory

/**
  * Created by saif on 06/07/17.
  */

class ChunkAssembler(override val uid: String) extends RawAnnotator[ChunkAssembler]{

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val annotatorType: AnnotatorType = CHUNK

  override val requiredAnnotatorTypes: Array[String] = Array(DOCUMENT)

  private val logger = LoggerFactory.getLogger("ChunkAssembler")

  val chunkCol = new Param(this, "chunkCol", "column that contains string. Must be part of DOCUMENT")
  val isArray = new BooleanParam(this, "isArray", "whether the chunkCol is an array of strings")

  setDefault(isArray -> false)

  def setChunkCol(value: String): this.type = set("chunkCol", value)

  def this() = this(Identifiable.randomUID("CHUNK_ASSEMBLER"))

  private def assemble = udf {
    (annotationProperties: Seq[Row], chunk: String) =>
      val annotations = annotationProperties.map(Annotation(_))
      annotations.flatMap(annotation => {
        val beginning = annotation.result.indexOf(chunk)
        val ending = beginning + chunk.length - 1
        if (beginning == -1) {
          logger.warn(s"Cannot proceed to assemble CHUNK, because could not find: `$chunk` within: `${annotation.result}`")
          None
        } else {
          Some(Annotation(
            annotatorType,
            beginning,
            ending,
            chunk,
            Map.empty[String, String]
          ))
        }
      })
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.withColumn($(outputCol), wrapColumnMetadata(assemble(col($(inputCols).head), col($(chunkCol)))))
  }

}
object ChunkAssembler extends DefaultParamsReadable[ChunkAssembler]
