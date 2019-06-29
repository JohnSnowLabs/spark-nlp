package com.johnsnowlabs.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
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
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol {

  import com.johnsnowlabs.nlp.AnnotatorType._

  private type DocumentationContent = Row

  val inputCol: Param[String] = new Param[String](this, "inputCol", "input text column for processing")

  val idCol: Param[String] = new Param[String](this, "idCol", "id column for row reference")

  val metadataCol: Param[String] = new Param[String](this, "metadataCol", "metadata for document column")

  /**
    * cleanupMode:
    * * disabled: keep original. Useful if need to head back to source later
    * * inplace: remove new lines and tabs, but not stringified, don't shrink
    * * inplace_full: remove new lines and tabs, including stringified, don't shrink
    * * shrink: remove new lines and tabs, but not stringified, do shrink
    * * shrink_full: remove new lines and tabs, stringified ones too, shrink all whitespaces
    */
  val cleanupMode: Param[String] = new Param[String](this, "cleanupMode", "possible values: disabled, inplace, inplace_full, shrink, shrink_full")

  setDefault(
    outputCol -> DOCUMENT,
    cleanupMode -> "disabled"
  )

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  def setInputCol(value: String): this.type = set(inputCol, value)

  def getInputCol: String = $(inputCol)

  def setIdCol(value: String): this.type = set(idCol, value)

  def getIdCol: String = $(idCol)

  def setMetadataCol(value: String): this.type = set(metadataCol, value)

  def getMetadataCol: String = $(metadataCol)

  def setCleanupMode(v: String): this.type = {
    v.trim.toLowerCase() match {
      case "disabled" => set(cleanupMode, "disabled")
      case "inplace" => set(cleanupMode, "inplace")
      case "inplace_full" => set(cleanupMode, "inplace_full")
      case "shrink" => set(cleanupMode, "shrink")
      case "shrink_full" => set(cleanupMode, "shrink_full")
      case b => throw new IllegalArgumentException(s"Special Character Cleanup supports only: disabled, inplace, inplace_full, shrink, shrink_full. Received: $b")
    }
  }

  def getCleanupMode: String = $(cleanupMode)

  def this() = this(Identifiable.randomUID("document"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  private[nlp] def assemble(text: String, metadata: Map[String, String]): Seq[Annotation] = {
    val possiblyCleaned = $(cleanupMode) match {
      case "disabled" => text
      case "inplace" => text.replaceAll("\\s", " ")
      case "inplace_full" => text.replaceAll("\\s|(?:\\\\r){0,1}(?:\\\\n)|(?:\\\\t)", " ")
      case "shrink" => text.trim.replaceAll("\\s+", " ")
      case "shrink_full" => text.trim.replaceAll("\\s+|(?:\\\\r)*(?:\\\\n)+|(?:\\\\t)+", " ")
      case b => throw new IllegalArgumentException(s"Special Character Cleanup supports only: disabled, inplace, inplace_full, shrink, shrink_full. Received: $b")
    }
    Seq(Annotation(outputAnnotatorType, 0, possiblyCleaned.length - 1, possiblyCleaned, metadata))
  }

  private[nlp] def assembleFromArray(texts: Seq[String]): Seq[Annotation] = {
    texts.zipWithIndex.flatMap{case (text, idx) =>
      assemble(text, Map("sentence" -> idx.toString))
    }
  }

  private def dfAssemble: UserDefinedFunction = udf {
    (text: String, id: String, metadata: Map[String, String]) =>
      assemble(text, metadata ++ Map("id" -> id, "sentence" -> "0"))
  }

  private def dfAssembleOnlyId: UserDefinedFunction = udf {
    (text: String, id: String) =>
      assemble(text, Map("id" -> id, "sentence" -> "0"))
  }

  private def dfAssembleNoId: UserDefinedFunction = udf {
    (text: String, metadata: Map[String, String]) =>
      assemble(text, metadata ++ Map("sentence" -> "0"))
  }

  private def dfAssembleNoExtras: UserDefinedFunction = udf {
    text: String =>
      assemble(text, Map("sentence" -> "0"))
  }

  private def dfAssemblyFromArray: UserDefinedFunction = udf {
    texts: Seq[String] => assembleFromArray(texts)
  }

  /** requirement for pipeline transformation validation. It is called on fit() */
  override final def transformSchema(schema: StructType): StructType = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    val outputFields = schema.fields :+
      StructField(getOutputCol, ArrayType(Annotation.dataType), nullable = false, metadataBuilder.build)
    StructType(outputFields)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    val documentAnnotations =
      if (dataset.schema.fields.find(_.name == getInputCol)
          .getOrElse(throw new IllegalArgumentException(s"Dataset does not have any '$getInputCol' column"))
          .dataType == ArrayType(StringType, containsNull = false))
        dfAssemblyFromArray(
          dataset.col(getInputCol)
        )
      else if (get(idCol).isDefined && get(metadataCol).isDefined)
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

