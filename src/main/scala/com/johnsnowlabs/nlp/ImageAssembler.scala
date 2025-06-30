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
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, regexp_replace, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

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

  /** Input text column for processing
    *
    * @group param
    */
  val textCol: Param[String] =
    new Param[String](this, "textCol", "input text column for processing")

  /** Input text column for processing
    *
    * @group setParam
    */
  def setTextCol(value: String): this.type = set(textCol, value)

  /** Input text column for processing
    *
    * @group getParam
    */
  def getTextCol: String = $(textCol)

  setDefault(inputCol -> IMAGE, outputCol -> "image_assembler", textCol -> "text")

  def this() = this(Identifiable.randomUID("ImageAssembler"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  private[nlp] def assemble(
      image: Option[ImageFields],
      metadata: Map[String, String],
      text: Option[String] = None): Seq[AnnotationImage] = {

    if (image.isDefined) {
      Seq(
        AnnotationImage(
          annotatorType = outputAnnotatorType,
          origin = image.get.origin,
          height = image.get.height,
          width = image.get.width,
          nChannels = image.get.nChannels,
          mode = image.get.mode,
          result = image.get.data,
          metadata = metadata,
          text = text.getOrElse("")))
    } else if (text.isDefined) {
      Seq(
        AnnotationImage(
          annotatorType = outputAnnotatorType,
          origin = "",
          height = 0,
          width = 0,
          nChannels = 0,
          mode = 0,
          result = Array.emptyByteArray,
          metadata = metadata,
          text = text.getOrElse("")))
    } else {
      Seq.empty[AnnotationImage]
    }

  }

  private[nlp] def dfAssemble: UserDefinedFunction = udf { (image: ImageFields) =>
    // Apache Spark has only 1 image per row
    assemble(Some(image), Map("image" -> "0"), None)
  }

  private[nlp] def dfAssembleWithText: UserDefinedFunction = udf {
    (image: ImageFields, text: String) =>
      // Apache Spark has only 1 image per row
      assemble(Some(image), Map("image" -> "0"), Some(text))
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

    val textColExists = dataset.schema.fields.exists(_.name == getTextCol)
    val imageAnnotations = if (textColExists) {
      dfAssembleWithText(dataset.col($(inputCol)), dataset.col($(textCol)))
    } else {
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

/** This is the companion object of [[ImageAssembler]]. Please refer to that class for the
  * documentation.
  */
object ImageAssembler extends DefaultParamsReadable[ImageAssembler] {

  /** Helper function that loads images from a path and returns them as raw bytes, instead of the
    * default OpenCV compatible format.
    *
    * Supported image types are JPEG, PNG, GIF, BMP (limited to images supported by stb_image.h).
    *
    * Multimodal inference with llama.cpp requires raw bytes as input.
    *
    * @param spark
    *   The SparkSession
    * @param path
    *   The path to the images. Supported image types are JPEG, PNG, GIF, BMP.
    * @return
    *   A dataframe with the images as raw bytes, as well as their metadata.
    */
  def loadImagesAsBytes(spark: SparkSession, path: String): DataFrame = {
    // Replace the path separator in the `origin` field and `path` column, so that they match
    def replacePath(columnName: String) = regexp_replace(col(columnName), ":///", ":/")

    val data: DataFrame =
      spark.read
        .format("image")
        .option("dropInvalid", value = true)
        .load(path)
        .withColumn("image", col("image").withField("origin", replacePath("image.origin")))

    val imageBytes: DataFrame =
      spark.read
        .format("binaryFile")
        .option("pathGlobFilter", "*.{jpeg,jpg,png,gif,bmp,JPEG,JPG,PNG,GIF,BMP}")
        .option("dropInvalid", value = true)
        .load(path)
        .withColumn("path", replacePath("path"))

    // Join on path
    val dfJoined =
      data.join(imageBytes, data("image.origin") === imageBytes("path"), "inner")

    // Replace image column data with image bytes
    val dfImageReplaced =
      dfJoined.withColumn("image", dfJoined("image").withField("data", dfJoined("content")))

    dfImageReplaced
  }
}
