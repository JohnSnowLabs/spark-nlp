/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/** Prepares data into a format that is processable by Spark NLP. This is the entry point for
  * every Spark NLP pipeline. The `DocumentAssembler` can read either a `String` column or an
  * `Array[String]`. Additionally, [[setCleanupMode]] can be used to pre-process the text
  * (Default: `disabled`). For possible options please refer the parameters section.
  *
  * For more extended examples on document pre-processing see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-assembler/Loading_Documents_With_DocumentAssembler.ipynb Examples]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  *
  * val data = Seq("Spark NLP is an open-source text processing library.").toDF("text")
  * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
  *
  * val result = documentAssembler.transform(data)
  *
  * result.select("document").show(false)
  * +----------------------------------------------------------------------------------------------+
  * |document                                                                                      |
  * +----------------------------------------------------------------------------------------------+
  * |[[document, 0, 51, Spark NLP is an open-source text processing library., [sentence -> 0], []]]|
  * +----------------------------------------------------------------------------------------------+
  *
  * result.select("document").printSchema
  * root
  *  |-- document: array (nullable = true)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- annotatorType: string (nullable = true)
  *  |    |    |-- begin: integer (nullable = false)
  *  |    |    |-- end: integer (nullable = false)
  *  |    |    |-- result: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  *  |    |    |-- embeddings: array (nullable = true)
  *  |    |    |    |-- element: float (containsNull = false)
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
class DocumentAssembler(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol {

  import com.johnsnowlabs.nlp.AnnotatorType._

  val EMPTY_STR = ""

  private type DocumentationContent = Row

  /** Input text column for processing
    *
    * @group param
    */
  val inputCol: Param[String] =
    new Param[String](this, "inputCol", "input text column for processing")

  /** Id column for row reference
    *
    * @group param
    */
  val idCol: Param[String] = new Param[String](this, "idCol", "id column for row reference")

  /** Metadata for document column
    *
    * @group param
    */
  val metadataCol: Param[String] =
    new Param[String](this, "metadataCol", "metadata for document column")

  /** cleanupMode can take the following values:
    *   - `disabled`: keep original. Useful if need to head back to source later
    *   - `inplace`: newlines and tabs into whitespaces, not stringified ones, don't trim
    *   - `inplace_full`: newlines and tabs into whitespaces, including stringified, don't trim
    *   - `shrink`: all whitespaces, newlines and tabs to a single whitespace, but not
    *     stringified, do trim
    *   - `shrink_full`: all whitespaces, newlines and tabs to a single whitespace, stringified
    *     ones too, trim all
    *   - `each`: newlines and tabs to one whitespace each
    *   - `each_full`: newlines and tabs, stringified ones too, to one whitespace each
    *   - `delete_full`: remove stringified newlines and tabs (replace with nothing)
    *
    * @group param
    */
  val cleanupMode: Param[String] = new Param[String](
    this,
    "cleanupMode",
    "possible values: " +
      "disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full")

  setDefault(outputCol -> DOCUMENT, cleanupMode -> "disabled")

  /** Output Annotator Type: DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** Input text column for processing
    *
    * @group setParam
    */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** Input text column for processing
    *
    * @group getParam
    */
  def getInputCol: String = $(inputCol)

  /** Id column for row reference
    *
    * @group setParam
    */
  def setIdCol(value: String): this.type = set(idCol, value)

  /** Id column for row reference
    *
    * @group getParam
    */
  def getIdCol: String = $(idCol)

  /** Metadata for document column
    *
    * @group setParam
    */
  def setMetadataCol(value: String): this.type = set(metadataCol, value)

  /** Metadata for document column
    *
    * @group getParam
    */
  def getMetadataCol: String = $(metadataCol)

  /** cleanupMode to pre-process text
    *
    * @group setParam
    */
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
      case b =>
        throw new IllegalArgumentException(s"Special Character Cleanup supports only: " +
          s"disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full. Received: $b")
    }
  }

  /** cleanupMode to pre-process text
    *
    * @group getParam
    */
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
      case b =>
        throw new IllegalArgumentException(s"Special Character Cleanup supports only: " +
          s"disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full. Received: $b")
    }
    try {
      Seq(
        Annotation(outputAnnotatorType, 0, possiblyCleaned.length - 1, possiblyCleaned, metadata))
    } catch {
      case _: Exception =>
        /*
         * when there is a null in the row
         * it outputs an empty Annotation
         * */
        Seq.empty[Annotation]
    }

  }

  private[nlp] def assembleFromArray(texts: Seq[String]): Seq[Annotation] = {
    texts.zipWithIndex.flatMap { case (text, idx) =>
      assemble(text, Map("sentence" -> idx.toString))
    }
  }

  private def dfAssemble: UserDefinedFunction = udf {
    (text: String, id: String, metadata: Map[String, String]) =>
      assemble(text, metadata ++ Map("id" -> id, "sentence" -> "0"))
  }

  private def dfAssembleOnlyId: UserDefinedFunction = udf { (text: String, id: String) =>
    assemble(text, Map("id" -> id, "sentence" -> "0"))
  }

  private def dfAssembleNoId: UserDefinedFunction = udf {
    (text: String, metadata: Map[String, String]) =>
      assemble(text, metadata ++ Map("sentence" -> "0"))
  }

  private def dfAssembleNoExtras: UserDefinedFunction = udf { text: String =>
    assemble(text, Map("sentence" -> "0"))
  }

  private def dfAssemblyFromArray: UserDefinedFunction = udf { texts: Seq[String] =>
    assembleFromArray(texts)
  }

  /** requirement for pipeline transformation validation. It is called on fit() */
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
    val documentAnnotations =
      if (dataset.schema.fields
          .find(_.name == getInputCol)
          .getOrElse(throw new IllegalArgumentException(
            s"Dataset does not have any '$getInputCol' column"))
          .dataType == ArrayType(StringType, containsNull = false))
        dfAssemblyFromArray(dataset.col(getInputCol))
      else if (get(idCol).isDefined && get(metadataCol).isDefined)
        dfAssemble(dataset.col(getInputCol), dataset.col(getIdCol), dataset.col(getMetadataCol))
      else if (get(idCol).isDefined)
        dfAssembleOnlyId(dataset.col(getInputCol), dataset.col(getIdCol))
      else if (get(metadataCol).isDefined)
        dfAssembleNoId(dataset.col(getInputCol), dataset.col(getMetadataCol))
      else
        dfAssembleNoExtras(dataset.col(getInputCol))
    dataset.withColumn(getOutputCol, documentAnnotations.as(getOutputCol, metadataBuilder.build))
  }

}

/** This is the companion object of [[DocumentAssembler]]. Please refer to that class for the
  * documentation.
  */
object DocumentAssembler extends DefaultParamsReadable[DocumentAssembler]
