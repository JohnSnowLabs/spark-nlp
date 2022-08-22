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

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.cv.util.schema.ImageSchemaUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

/** Prepares images read by Spark into a format that is processable by Spark NLP. This component
  * is needed to process images.
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.ImageAssembler
  * import org.apache.spark.ml.Pipeline
  *
  * val imageDF: DataFrame = spark.read
  *   .format("image")
  *   .option("dropInvalid", value = true)
  *   .load("src/test/resources/image/")
  *
  * val imageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val pipeline = new Pipeline().setStages(Array(imageAssembler))
  * val pipelineDF = pipeline.fit(imageDF).transform(imageDF)
  * pipelineDF.printSchema()
  * root
  *  |-- image_assembler: array (nullable = true)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- annotatorType: string (nullable = true)
  *  |    |    |-- origin: string (nullable = true)
  *  |    |    |-- height: integer (nullable = false)
  *  |    |    |-- width: integer (nullable = false)
  *  |    |    |-- nChannels: integer (nullable = false)
  *  |    |    |-- mode: integer (nullable = false)
  *  |    |    |-- result: binary (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
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
class ImageAssembler(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol {

  /** Output Annotator Type: DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = IMAGE

  /** Input text column for processing
    *
    * @group param
    */
  val inputCol: Param[String] =
    new Param[String](this, "inputCol", "input text column for processing")

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

  setDefault(inputCol -> IMAGE, outputCol -> "image_assembler")

  def this() = this(Identifiable.randomUID("ImageAssembler"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  private[nlp] def assemble(
      image: ImageFields,
      metadata: Map[String, String]): Seq[AnnotationImage] = {

    Seq(
      AnnotationImage(
        annotatorType = outputAnnotatorType,
        origin = image.origin,
        height = image.height,
        width = image.width,
        nChannels = image.nChannels,
        mode = image.mode,
        result = image.data,
        metadata = metadata))

  }

  private[nlp] def dfAssemble: UserDefinedFunction = udf { (image: ImageFields) =>
    // Apache Spark has only 1 image per row
    assemble(image, Map("image" -> "0"))
  }

  /** requirement for pipeline transformation validation. It is called on fit() */
  override final def transformSchema(schema: StructType): StructType = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    val outputFields = schema.fields :+
      StructField(
        getOutputCol,
        ArrayType(AnnotationImage.dataType),
        nullable = false,
        metadataBuilder.build)
    StructType(outputFields)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    require(
      dataset.schema.fields.exists(_.name == getInputCol),
      s"column $getInputCol is not presented in your DataFrame")
    require(
      ImageSchemaUtils.isImage(dataset.schema(getInputCol)),
      s"column $getInputCol doesn't have Apache Spark ImageSchema. Make sure you read your images via spark.read.format(image).load(PATH)")

    val imageAnnotations = {
      dfAssemble(dataset($(inputCol)))
    }

    dataset.withColumn(getOutputCol, imageAnnotations.as(getOutputCol, metadataBuilder.build))
  }

}

private[nlp] case class ImageFields(
    origin: String,
    height: Int,
    width: Int,
    nChannels: Int,
    mode: Int,
    data: Array[Byte])
