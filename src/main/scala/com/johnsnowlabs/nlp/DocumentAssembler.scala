package com.johnsnowlabs.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}


class DocumentAssembler(override val uid: String)
  extends Transformer
    with DefaultParamsWritable
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol {

  import com.johnsnowlabs.nlp.AnnotatorType._

  val EMPTY_STR = ""

  private type DocumentationContent = Row

  val inputCol: Param[String] = new Param[String](this, "inputCol", "input text column for processing")

  val idCol: Param[String] = new Param[String](this, "idCol", "id column for row reference")

  val metadataCol: Param[String] = new Param[String](this, "metadataCol", "metadata for document column")

  /**
    * cleanupMode:
    * * disabled: keep original. Useful if need to head back to source later
    * * inplace: newlines and tabs into whitespaces, not stringified ones, don't trim
    * * inplace_full: newlines and tabs into whitespaces, including stringified, don't trim
    * * shrink: all whitespaces, newlines and tabs to a single whitespace, but not stringified, do trim
    * * shrink_full: all whitespaces, newlines and tabs to a single whitespace, stringified ones too, trim all
    * * each: newlines and tabs to one whitespace each
    * * each_full: newlines and tabs, stringified ones too, to one whitespace each
    * * delete_full: remove stringified newlines and tabs (replace with nothing)
    */
  val cleanupMode: Param[String] = new Param[String](this, "cleanupMode", "possible values: " +
    "disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full")

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
      case "each" => set(cleanupMode, "each")
      case "each_full" => set(cleanupMode, "each_full")
      case "delete_full" => set(cleanupMode, "delete_full")
      case b => throw new IllegalArgumentException(s"Special Character Cleanup supports only: " +
        s"disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full. Received: $b")
    }
  }

  def getCleanupMode: String = $(cleanupMode)

  def this() = this(Identifiable.randomUID("document"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  private[nlp] def assemble(text: String, metadata: Map[String, String]): Seq[Annotation] = {

    val _text = Option(text).getOrElse(EMPTY_STR)

    val possiblyCleaned = $(cleanupMode) match {
      case "disabled" => _text
      case "inplace" => _text.replaceAll("\\s", " ")
      case "inplace_full" => _text.replaceAll("\\s|(?:\\\\r)?(?:\\\\n)|(?:\\\\t)", " ")
      case "shrink" => _text.trim.replaceAll("\\s+", " ")
      case "shrink_full" => _text.trim.replaceAll("\\s+|(?:\\\\r)*(?:\\\\n)+|(?:\\\\t)+", " ")
      case "each" => _text.replaceAll("\\s[\\n\\t]", " ")
      case "each_full" => _text.replaceAll("\\s(?:\\n|\\t|(?:\\\\r)?(?:\\\\n)|(?:\\\\t))", " ")
      case "delete_full" => _text.trim.replaceAll("(?:\\\\r)?(?:\\\\n)|(?:\\\\t)", "")
      case b => throw new IllegalArgumentException(s"Special Character Cleanup supports only: " +
        s"disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full. Received: $b")
    }
    try {
      Seq(Annotation(outputAnnotatorType, 0, possiblyCleaned.length - 1, possiblyCleaned, metadata))
    } catch { case _: Exception =>
      /*
      * when there is a null in the row
      * it outputs an empty Annotation
      * */
      Seq.empty[Annotation]
    }

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

