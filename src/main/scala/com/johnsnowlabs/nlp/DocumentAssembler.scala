package com.johnsnowlabs.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
  * Created by saif on 06/07/17.
  */

class DocumentAssembler(override val uid: String)
  extends Transformer
    with DefaultParamsWritable
    with HasAnnotatorType
    with HasOutputAnnotationCol {

  import com.johnsnowlabs.nlp.AnnotatorType._

  private type DocumentationContent = Row

  val inputCol: Param[String] = new Param[String](this, "inputCol", "input text column for processing")

  val idCol: Param[String] = new Param[String](this, "idCol", "id column for row reference")

  val metadataCol: Param[String] = new Param[String](this, "metadataCol", "metadata for document column")

  setDefault(outputCol, DOCUMENT)

  override val annotatorType: AnnotatorType = DOCUMENT

  def setInputCol(value: String): this.type = set(inputCol, value)

  def getInputCol: String = $(inputCol)

  def setIdCol(value: String): this.type = set(idCol, value)

  def getIdCol: String = $(idCol)

  def setMetadataCol(value: String): this.type = set(metadataCol, value)

  def getMetadataCol: String = $(metadataCol)

  def this() = this(Identifiable.randomUID("document"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  private def assemble(text: String, metadata: Map[String, String]): Seq[Annotation] = {
    Seq(Annotation(
      annotatorType,
      text,
      metadata ++ Map(Annotation.BEGIN -> "0", Annotation.END -> (text.length - 1).toString))
    )
  }

  private def dfAssemble: UserDefinedFunction = udf {
    (text: String, id: String, metadata: Map[String, String]) =>
      assemble(text, metadata ++ Map(Annotation.BEGIN -> "0", Annotation.END -> (text.length - 1).toString, "id" -> id))
  }

  private def dfAssembleOnlyId: UserDefinedFunction = udf {
    (text: String, id: String) =>
      assemble(text, Map(Annotation.BEGIN -> "0", Annotation.END -> (text.length - 1).toString, "id" -> id))
  }

  private def dfAssembleNoId: UserDefinedFunction = udf {
    (text: String, metadata: Map[String, String]) =>
      assemble(text, metadata ++ Map(Annotation.BEGIN -> "0", Annotation.END -> (text.length - 1).toString))
  }

  private def dfAssembleNoExtras: UserDefinedFunction = udf {
    (text: String) =>
      assemble(text, Map(Annotation.BEGIN -> "0", Annotation.END -> (text.length - 1).toString))
  }

  /** requirement for pipeline transformation validation. It is called on fit() */
  override final def transformSchema(schema: StructType): StructType = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)
    val outputFields = schema.fields :+
      StructField(getOutputCol, ArrayType(Annotation.dataType), nullable = false, metadataBuilder.build)
    StructType(outputFields)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)
    val documentAnnotations =
      if (get(idCol).isDefined && get(metadataCol).isDefined)
        dfAssemble(
          dataset.col(getInputCol),
          dataset.col(getIdCol),
          dataset.col(getMetadataCol)
        )
      else if (get(idCol).isDefined)
        dfAssembleOnlyId(
          dataset.col(getInputCol),
          dataset.col(getIdCol)
        )
      else if (get(metadataCol).isDefined)
        dfAssembleNoId(
          dataset.col(getInputCol),
          dataset.col(getMetadataCol)
        )
      else
        dfAssembleNoExtras(
          dataset.col(getInputCol)
        )
    dataset.withColumn(
      getOutputCol,
      documentAnnotations.as(getOutputCol, metadataBuilder.build)
    )
  }

}
object DocumentAssembler extends DefaultParamsReadable[DocumentAssembler]