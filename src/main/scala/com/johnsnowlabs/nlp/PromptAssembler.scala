package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.llama.LlamaModel
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/** TODO: Add description
  *
  * @param uid
  */
class PromptAssembler(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol {
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  def this() = this(Identifiable.randomUID("PROMPT_ASSEMBLER"))

  val chatTemplate: Param[String] =
    new Param[String](this, "chatTemplate", "Template used for the chat")

  val inputCol: Param[String] =
    new Param[String](this, "inputCol", "Input column containing a sequence of messages")

  val addAssistant: BooleanParam =
    new BooleanParam(
      this,
      "addAssistant",
      "Whether to add an assistant header to the end of the generated string")

  setDefault(addAssistant -> true)

  /** Sets the input text column for processing
    *
    * @group setParam
    */
  def setInputCol(value: String): this.type = set(inputCol, value)
  def getInputCol: String = $(inputCol)

  /** Sets the chat template to be used for the chat. Should be a Jinja2 template or something
    * similar llama.cpp can parse.
    *
    * @param value
    *   The template to use
    */
  def setChatTemplate(value: String): this.type = set(chatTemplate, value)

  /** Gets the chat template to be used for the chat.
    *
    * @return
    *   The template to use
    */
  def getChatTemplate: String = $(chatTemplate)

  /** Whether to add an assistant header to the end of the generated string.
    *
    * @param value
    *   Whether to add the assistant header
    */
  def setAddAssistant(value: Boolean): this.type = set(addAssistant, value)

  /** Whether to add an assistant header to the end of the generated string.
    *
    * @return
    *   Whether to add the assistant header
    */
  def getAddAssistant: Boolean = $(addAssistant)

  // Expected Input type of the input column
  private val expectedInputType = ArrayType(
    StructType(
      Seq(
        StructField("_1", StringType, nullable = true),
        StructField("_2", StringType, nullable = true))),
    containsNull = true)

  /** Adds the result Annotation type to the schema.
    *
    * Requirement for pipeline transformation validation. It is called on fit()
    */
  override final def transformSchema(schema: StructType): StructType = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    val outputFields = schema.fields :+
      StructField(
        getOutputCol,
        ArrayType(Annotation.dataType),
        nullable = false,
        metadataBuilder.build)
    StructType(outputFields)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    val columnDataType = dataset.schema.fields
      .find(_.name == getInputCol)
      .getOrElse(
        throw new IllegalArgumentException(s"Dataset does not have any '$getInputCol' column"))
      .dataType

    val documentAnnotations: Column =
      if (columnDataType == expectedInputType) applyTemplate(dataset.col(getInputCol))
      else
        throw new IllegalArgumentException(
          s"Column '$getInputCol' must be of type $expectedInputType, but got $columnDataType")

    dataset.withColumn(getOutputCol, documentAnnotations.as(getOutputCol, metadataBuilder.build))
  }

  private def applyTemplate: UserDefinedFunction = udf { chat: Seq[(String, String)] =>
    val template = $(chatTemplate)

    val chatArray = chat.map { case (role, text) =>
      Array(role, text)
    }.toArray

    val chatString = LlamaModel.applyChatTemplate(template, chatArray, $(addAssistant))
    Annotation(chatString)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}
