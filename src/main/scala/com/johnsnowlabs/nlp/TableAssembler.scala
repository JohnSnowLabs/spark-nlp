package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TABLE}
import com.johnsnowlabs.nlp.annotators.common.TableData
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class TableAssembler(override val uid: String)
    extends AnnotatorModel[TokenAssembler]
    with HasSimpleAnnotate[TokenAssembler] {

  def this() = this(Identifiable.randomUID("TableAssembler"))

  /** Output annotator types: DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TABLE

  /** Input annotator types: DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  protected val INPUT_FORMATS = Array("json", "csv")

  val inputFormat = new Param[String](
    this,
    "inputFormat",
    "Input table format. Supported formats: %s".format(INPUT_FORMATS.mkString(", ")))

  def setInputType(value: String): this.type = {
    require(
      INPUT_FORMATS.contains(value),
      "Invalid input format. Currently supported formats are: " + INPUT_FORMATS.mkString(", "))
    set(this.inputFormat, value.toLowerCase)
  }

  def getInputType: String = $(inputFormat)

  val csvDelimiter = new Param[String](this, "csvDelimiter", "CSV delimiter")

  def setCsvDelimiter(value: String): this.type = {
    set(this.csvDelimiter, value)
  }

  def getCsvDelimiter: String = $(csvDelimiter)

  setDefault(inputFormat -> "json", csvDelimiter -> ",")

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations
      .map { annotation =>
        val jsonTable = $(inputFormat) match {
          case "json" => TableData.fromJson(annotation.result).toJson
          case "csv" => TableData.fromCsv(annotation.result, $(csvDelimiter)).toJson
        }
        new Annotation(
          annotatorType = TABLE,
          begin = 0,
          end = jsonTable.length,
          result = jsonTable,
          metadata = annotation.metadata ++ Map("original_input_type" -> $(inputFormat)))
      }

  }
}

/** This is the companion object of [[TableAssembler]]. Please refer to that class for the
  * documentation.
  */
object TableAssembler extends DefaultParamsReadable[DocumentAssembler]
