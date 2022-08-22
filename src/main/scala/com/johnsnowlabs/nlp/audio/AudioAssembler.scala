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

package com.johnsnowlabs.nlp.audio

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.{HasOutputAnnotationCol, HasOutputAnnotatorType}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

/** Prepares audio read by Spark into a format that is processable by Spark NLP. This component is
  * needed to process audio.
  *
  * TODO:
  *   - support various file formats
  *   - output into float array
  *   - handle stereo
  */
class AudioAssembler(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol {

  /** Output Annotator Type: AUDIO
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AUDIO

  /** Input column of raw bytes of the audio
    *
    * @group param
    */
  val inputCol: Param[String] =
    new Param[String](this, "inputCol", "Input column of raw bytes of the audio")

  /** Input column of raw audio
    *
    * @group setParam
    */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** Input column of raw audio
    *
    * @group getParam
    */
  def getInputCol: String = $(inputCol)

  setDefault(inputCol -> AUDIO, outputCol -> "audio_assembler")

  def this() = this(Identifiable.randomUID("AudioAssembler"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  private[nlp] def assemble(rawAudio: Array[Byte]): Seq[AnnotationAudio] = {

    Seq(AnnotationAudio(rawAudio))
  }

  private[nlp] def dfAssemble: UserDefinedFunction = udf { (rawAudio: Array[Byte]) =>
    assemble(rawAudio)
  }

  /** requirement for pipeline transformation validation. It is called on fit() */
  override final def transformSchema(schema: StructType): StructType = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    val outputFields = schema.fields :+
      StructField(
        getOutputCol,
        ArrayType(AnnotationAudio.dataType),
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

//    TODO: Check valid file individually?
//    require(
//      ImageSchemaUtils.isImage(dataset.schema(getInputCol)),
//      s"column $getInputCol doesn't have Apache Spark ImageSchema. Make sure you read your images via spark.read.format(image).load(PATH)")
//
    val audioAnnotations = {
      dfAssemble(dataset($(inputCol)))
    }
//
    dataset.withColumn(getOutputCol, audioAnnotations.as(getOutputCol, metadataBuilder.build))

  }

}
