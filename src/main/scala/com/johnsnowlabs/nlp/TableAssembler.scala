package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TABLE}
import com.johnsnowlabs.nlp.annotators.common.TableData
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.json4s.jackson.JsonMethods.parse
import com.johnsnowlabs.util.JsonParser.formats

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

  protected val INPUT_TYPES = Array("json", "csv")

  val inputType = new Param[String](this, "inputType", "Input table type. Supported types: %s".format(INPUT_TYPES.mkString(", ")))

  def setInputType(value: String): this.type = {
    require(INPUT_TYPES.contains(value), "Invalid input type. Currently supported input types are: " + INPUT_TYPES.mkString(", "))
    set(this.inputType, value.toLowerCase)
  }

  def getInputType: String = $(inputType)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {


    annotations
      .map{
        annotation =>
          val jsonTable = $(inputType) match {
            case "json" =>
              try{
                parse(annotation.result).extract[TableData]
                annotation.result
              } catch  {
                case _: Exception => throw new Exception("Invalid JSON input")
              }
            case "csv" => "a"
            case _ => throw new Exception("Unsupported input type: %s".format($(inputType)))
          }
          new Annotation(
            annotatorType = TABLE,
            begin = 0,
            end = jsonTable.length,
            result = jsonTable,
            metadata = annotation.metadata ++ Map(
              "original_input_type" -> $(inputType)
            ))
      }

  }
}
