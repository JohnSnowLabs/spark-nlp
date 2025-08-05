package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.llama.LlamaExtensions
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/** Assembles a sequence of messages into a single string using a template. These strings can then
  * be used as prompts for large language models.
  *
  * This annotator expects an array of two-tuples as the type of the input column (one array of
  * tuples per row). The first element of the tuples should be the role and the second element is
  * the text of the message. Possible roles are "system", "user" and "assistant".
  *
  * An assistant header can be added to the end of the generated string by using
  * `setAddAssistant(true)`.
  *
  * At the moment, this annotator uses llama.cpp as a backend to parse and apply the templates.
  * llama.cpp uses basic pattern matching to determine the type of the template, then applies a
  * basic version of the template to the messages. This means that more advanced templates are not
  * supported.
  *
  * For an extended example see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/llama.cpp/PromptAssember_with_AutoGGUFModel.ipynb example notebook]].
  *
  * ==Example==
  * {{{
  * // Batches (whole conversations) of arrays of messages
  * val data: Seq[Seq[(String, String)]] = Seq(
  *   Seq(
  *     ("system", "You are a helpful assistant."),
  *     ("assistant", "Hello there, how can I help you?"),
  *     ("user", "I need help with organizing my room.")))
  *
  * val dataDF = data.toDF("messages")
  *
  * // llama3.1
  * val template =
  *   "{{- bos_token }} {%- if custom_tools is defined %} {%- set tools = custom_tools %} {%- " +
  *     "endif %} {%- if not tools_in_user_message is defined %} {%- set tools_in_user_message = true %} {%- " +
  *     "endif %} {%- if not date_string is defined %} {%- set date_string = \"26 Jul 2024\" %} {%- endif %} " +
  *     "{%- if not tools is defined %} {%- set tools = none %} {%- endif %} {#- This block extracts the " +
  *     "system message, so we can slot it into the right place. #} {%- if messages[0]['role'] == 'system' %}" +
  *     " {%- set system_message = messages[0]['content']|trim %} {%- set messages = messages[1:] %} {%- else" +
  *     " %} {%- set system_message = \"\" %} {%- endif %} {#- System message + builtin tools #} {{- " +
  *     "\"<|start_header_id|>system<|end_header_id|>\\n\\n\" }} {%- if builtin_tools is defined or tools is " +
  *     "not none %} {{- \"Environment: ipython\\n\" }} {%- endif %} {%- if builtin_tools is defined %} {{- " +
  *     "\"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}} " +
  *     "{%- endif %} {{- \"Cutting Knowledge Date: December 2023\\n\" }} {{- \"Today Date: \" + date_string " +
  *     "+ \"\\n\\n\" }} {%- if tools is not none and not tools_in_user_message %} {{- \"You have access to " +
  *     "the following functions. To call a function, please respond with JSON for a function call.\" }} {{- " +
  *     "'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its" +
  *     " value}.' }} {{- \"Do not use variables.\\n\\n\" }} {%- for t in tools %} {{- t | tojson(indent=4) " +
  *     "}} {{- \"\\n\\n\" }} {%- endfor %} {%- endif %} {{- system_message }} {{- \"<|eot_id|>\" }} {#- " +
  *     "Custom tools are passed in a user message with some extra guidance #} {%- if tools_in_user_message " +
  *     "and not tools is none %} {#- Extract the first user message so we can plug it in here #} {%- if " +
  *     "messages | length != 0 %} {%- set first_user_message = messages[0]['content']|trim %} {%- set " +
  *     "messages = messages[1:] %} {%- else %} {{- raise_exception(\"Cannot put tools in the first user " +
  *     "message when there's no first user message!\") }} {%- endif %} {{- " +
  *     "'<|start_header_id|>user<|end_header_id|>\\n\\n' -}} {{- \"Given the following functions, please " +
  *     "respond with a JSON for a function call \" }} {{- \"with its proper arguments that best answers the " +
  *     "given prompt.\\n\\n\" }} {{- 'Respond in the format {\"name\": function name, \"parameters\": " +
  *     "dictionary of argument name and its value}.' }} {{- \"Do not use variables.\\n\\n\" }} {%- for t in " +
  *     "tools %} {{- t | tojson(indent=4) }} {{- \"\\n\\n\" }} {%- endfor %} {{- first_user_message + " +
  *     "\"<|eot_id|>\"}} {%- endif %} {%- for message in messages %} {%- if not (message.role == 'ipython' " +
  *     "or message.role == 'tool' or 'tool_calls' in message) %} {{- '<|start_header_id|>' + message['role']" +
  *     " + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }} {%- elif 'tool_calls' in " +
  *     "message %} {%- if not message.tool_calls|length == 1 %} {{- raise_exception(\"This model only " +
  *     "supports single tool-calls at once!\") }} {%- endif %} {%- set tool_call = message.tool_calls[0]" +
  *     ".function %} {%- if builtin_tools is defined and tool_call.name in builtin_tools %} {{- " +
  *     "'<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}} {{- \"<|python_tag|>\" + tool_call.name + " +
  *     "\".call(\" }} {%- for arg_name, arg_val in tool_call.arguments | items %} {{- arg_name + '=\"' + " +
  *     "arg_val + '\"' }} {%- if not loop.last %} {{- \", \" }} {%- endif %} {%- endfor %} {{- \")\" }} {%- " +
  *     "else %} {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}} {{- '{\"name\": \"' + " +
  *     "tool_call.name + '\", ' }} {{- '\"parameters\": ' }} {{- tool_call.arguments | tojson }} {{- \"}\" " +
  *     "}} {%- endif %} {%- if builtin_tools is defined %} {#- This means we're in ipython mode #} {{- " +
  *     "\"<|eom_id|>\" }} {%- else %} {{- \"<|eot_id|>\" }} {%- endif %} {%- elif message.role == \"tool\" " +
  *     "or message.role == \"ipython\" %} {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }} {%- " +
  *     "if message.content is mapping or message.content is iterable %} {{- message.content | tojson }} {%- " +
  *     "else %} {{- message.content }} {%- endif %} {{- \"<|eot_id|>\" }} {%- endif %} {%- endfor %} {%- if " +
  *     "add_generation_prompt %} {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }} {%- endif %} "
  *
  * val promptAssembler = new PromptAssembler()
  *   .setInputCol("messages")
  *   .setOutputCol("prompt")
  *   .setChatTemplate(template)
  *
  * promptAssembler.transform(dataDF).select("prompt.result").show(truncate = false)
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |result                                                                                                                                                                                                                                                                                                                      |
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello there, how can I help you?<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI need help with organizing my room.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n]|
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  *
  * }}}
  *
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

  /** Sets the chat template to be used for the chat. Should be something that llama.cpp can
    * parse.
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
          s"Column '$getInputCol' must be of type Array[(String, String)] " +
            s"(exactly '$expectedInputType'), but got $columnDataType")

    dataset.withColumn(getOutputCol, documentAnnotations.as(getOutputCol, metadataBuilder.build))
  }

  private def applyTemplate: UserDefinedFunction = udf { chat: Seq[(String, String)] =>
    try {
      val template = $(chatTemplate)

      val chatArray = chat.map { case (role, text) =>
        Array(role, text)
      }.toArray

      val chatString = LlamaExtensions.applyChatTemplate(template, chatArray, $(addAssistant))
      Seq(Annotation(chatString))
    } catch {
      case _: Exception =>
        /*
         * when there is a null in the row
         * it outputs an empty Annotation
         * */
        Seq.empty
    }
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}

/** This is the companion object of [[PromptAssembler]]. Please refer to that class for the
  * documentation.
  */
object PromptAssembler extends DefaultParamsReadable[PromptAssembler]
