package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TABLE}
import com.johnsnowlabs.nlp.annotators.common.TableData
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/** This transformer parses text into tabular representation. The input consists of DOCUMENT
  * annotations and the output are TABLE annotations. The source format can be either JSON or CSV.
  * The format of the JSON files should be:
  *
  * {{{
  * {
  *   "header": [col1, col2, ..., colN],
  *   "rows": [
  *     [val11, val12, ..., val1N],
  *     [val22, va22, ..., val2N],
  *     ...
  *    ]
  * }
  * }}}
  * The CSV format support alternative delimiters (e.g. tab), as well as escaping delimiters by
  * surrounding cell values with double quotes. For example:
  *
  * {{{
  * column1, column2, "column with, comma"
  * value1, value2, value3
  * "escaped value", "value with, comma", "value with double ("") quote"
  * }}}
  *
  * The transformer stores tabular data internally as JSON. The default input format is also JSON.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import org.apache.spark.ml.Pipeline
  *
  * val csvData =
  *   """
  *     |"name", "money", "age"
  *     |"Donald Trump", "$100,000,000", "75"
  *     |"Elon Musk", "$20,000,000,000,000", "55"
  *     |""".stripMargin.trim
  *
  * val data =Seq(csvData).toDF("csv")
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("csv")
  *   .setOutputCol("document")
  *
  * val tableAssembler = new TableAssembler()
  *   .setInputCols(Array("document"))
  *   .setOutputCol("table")
  *   .setInputFormat("csv")
  *
  * val pipeline = new Pipeline()
  *   .setStages(
  *     Array(documentAssembler, tableAssembler)
  *     ).fit(data)
  *
  * val result = pipeline.transform(data)
  * result
  *   .selectExpr("explode(table) AS table")
  *   .select("table.result", "table.metadata.input_format")
  *   .show(false)
  *
  * +--------------------------------------------+-------------+
  * |result                                      |input_format |
  * +--------------------------------------------+-------------+
  * |{                                           |csv          |
  * | "header": ["name","money","age"],          |             |
  * |  "rows":[                                  |             |
  * |   ["Donald Trump","$100,000,000","75"],    |             |
  * |   ["Elon Musk","$20,000,000,000,000","55"] |             |
  * |  ]                                         |             |
  * |}                                           |             |
  * +--------------------------------------------+-------------+
  * }}}
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

  def setInputFormat(value: String): this.type = {
    require(
      INPUT_FORMATS.contains(value),
      "Invalid input format. Currently supported formats are: " + INPUT_FORMATS.mkString(", "))
    set(this.inputFormat, value.toLowerCase)
  }

  def getInputFormat: String = $(inputFormat)

  val csvDelimiter = new Param[String](this, "csvDelimiter", "CSV delimiter")

  def setCsvDelimiter(value: String): this.type = {
    set(this.csvDelimiter, value)
  }

  def getCsvDelimiter: String = $(csvDelimiter)

  val escapeCsvDelimiter =
    new BooleanParam(
      this,
      "escapeCsvDelimiter",
      "Escape Csv delimiter by surrounding values with double quotes")

  def setEscapeCsvDelimiter(value: Boolean): this.type = {
    set(this.escapeCsvDelimiter, value)
  }

  def getEscapeCsvDelimiter: Boolean = $(escapeCsvDelimiter)

  setDefault(inputFormat -> "json", csvDelimiter -> ",", escapeCsvDelimiter -> true)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations
      .map { annotation =>
        val jsonTable = $(inputFormat) match {
          case "json" => TableData.fromJson(annotation.result).toJson
          case "csv" =>
            TableData.fromCsv(annotation.result, $(csvDelimiter), $(escapeCsvDelimiter)).toJson
        }
        new Annotation(
          annotatorType = TABLE,
          begin = 0,
          end = jsonTable.length,
          result = jsonTable,
          metadata = annotation.metadata ++ Map("input_format" -> $(inputFormat)))
      }

  }
}

/** This is the companion object of [[TableAssembler]]. Please refer to that class for the
  * documentation.
  */
object TableAssembler extends DefaultParamsReadable[DocumentAssembler]
