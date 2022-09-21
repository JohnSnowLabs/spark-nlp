package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TABLE}
import com.johnsnowlabs.nlp.annotators.common.TableData
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable


class TableAssembler (override val uid: String)
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

  val inputFormat = new Param[String](this, "inputFormat", "Input table format. Supported formats: %s".format(INPUT_FORMATS.mkString(", ")))

  def setInputType(value: String): this.type = {
    require(INPUT_FORMATS.contains(value), "Invalid input format. Currently supported formats are: " + INPUT_FORMATS.mkString(", "))
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
      .map{
        annotation =>
          val jsonTable = try{
            $(inputFormat) match {
              case "json" => TableData.fromJSON(annotation.result).toJSON
              case "csv" => TableData.fromCSV(annotation.result, $(csvDelimiter)).toJSON
            }
          } catch {
            case _: Exception => throw new Exception("Invalid %s input".format($(inputFormat)))
          }

          new Annotation(
            annotatorType = TABLE,
            begin = 0,
            end = jsonTable.length,
            result = jsonTable,
            metadata = annotation.metadata ++ Map(
              "original_input_type" -> $(inputFormat)
            ))
      }

  }
}
