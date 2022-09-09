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
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

/** Prepares audio read by Spark into a format that is processable by Spark NLP. This component is
  * needed to process audio.
  *
  * Input col is a single record that contains the raw content and metadata of the file.
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.AudioAssembler
  * import org.apache.spark.ml.Pipeline
  *
  * val audioAssembler = new AudioAssembler()
  *   .setInputCol("audio")
  *   .setOutputCol("audio_assembler")
  *
  * val pipeline = new Pipeline().setStages(Array(audioAssembler))
  * val pipelineDF = pipeline.fit(imageDF).transform(wavDf)
  * pipelineDF.printSchema()
  * root
  *
  * }}}
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

  /** Input column of raw floats of the processed audio
    *
    * @group param
    */
  val inputCol: Param[String] =
    new Param[String](this, "inputCol", "Input column of raw floats of the audio")

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

  private[nlp] def assemble(
      audio: Array[Float],
      metadata: Map[String, String]): Seq[AnnotationAudio] = {

    Seq(
      new AnnotationAudio(
        AnnotatorType.AUDIO,
        result = audio,
        metadata = Map("length" -> audio.length.toString) ++ metadata))
  }

  private[nlp] def dfAssemble: UserDefinedFunction = udf { (audio: Array[Float]) =>
    assemble(audio, Map("audio" -> "0"))
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

    val inputColSchema = dataset.schema(getInputCol).dataType

    require(
      inputColSchema == ArrayType(FloatType),
      s"""column $getInputCol is not of ArrayType(FloatType) type. Instead it is $inputColSchema. Please make sure your inputCol contains Array[Float]""")

    val audioAnnotations = {
      dfAssemble(dataset($(inputCol)))
    }

    dataset.withColumn(getOutputCol, audioAnnotations.as(getOutputCol, metadataBuilder.build))

  }

}

private[nlp] case class AudioFields(data: Array[Float])

/** This is the companion object of [[AudioAssembler]]. Please refer to that class for the
  * documentation.
  */
object AudioAssembler extends DefaultParamsReadable[AudioAssembler]
