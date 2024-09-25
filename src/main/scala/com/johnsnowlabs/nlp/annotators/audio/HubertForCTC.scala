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

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel}
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowWrapper}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  detectEngine,
  loadJsonStringAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.Preprocessor
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** Hubert Model with a language modeling head on top for Connectionist Temporal Classification
  * (CTC). Hubert was proposed in HuBERT: Self-Supervised Speech Representation Learning by Masked
  * Prediction of Hidden Units by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal
  * Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.
  *
  * The annotator takes audio files and transcribes it as text. The audio needs to be provided
  * pre-processed an array of floats.
  *
  * Note that this annotator is currently not supported on Apple Silicon processors such as the
  * M1/M2 (Apple Silicon). This is due to the processor not supporting instructions for XLA.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val speechToText = HubertForCTC.pretrained()
  *   .setInputCols("audio_assembler")
  *   .setOutputCol("text")
  * }}}
  * The default model is `"asr_hubert_large_ls960"`, if no name is provided.
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/audio/HubertForCTCTestSpec.scala HubertForCTCTestSpec]].
  *
  * '''References:'''
  *
  * [[https://arxiv.org/abs/2106.07447 HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units]]
  *
  * '''Paper Abstract:'''
  *
  * ''Self-supervised approaches for speech representation learning are challenged by three unique
  * problems: (1) there are multiple sound units in each input utterance, (2) there is no lexicon
  * of input sound units during the pre-training phase, and (3) sound units have variable lengths
  * with no explicit segmentation. To deal with these three problems, we propose the Hidden-Unit
  * BERT (HuBERT) approach for self-supervised speech representation learning, which utilizes an
  * offline clustering step to provide aligned target labels for a BERT-like prediction loss. A
  * key ingredient of our approach is applying the prediction loss over the masked regions only,
  * which forces the model to learn a combined acoustic and language model over the continuous
  * inputs. HuBERT relies primarily on the consistency of the unsupervised clustering step rather
  * than the intrinsic quality of the assigned cluster labels. Starting with a simple k-means
  * teacher of 100 clusters, and using two iterations of clustering, the HuBERT model either
  * matches or improves upon the state-of-the-art wav2vec 2.0 performance on the Librispeech
  * (960h) and Libri-light (60,000h) benchmarks with 10min, 1h, 10h, 100h, and 960h fine-tuning
  * subsets. Using a 1B parameter model, HuBERT shows up to 19% and 13% relative WER reduction on
  * the more challenging dev-other and test-other evaluation subsets.''
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotators._
  * import com.johnsnowlabs.nlp.annotators.audio.HubertForCTC
  * import org.apache.spark.ml.Pipeline
  *
  * val audioAssembler: AudioAssembler = new AudioAssembler()
  *   .setInputCol("audio_content")
  *   .setOutputCol("audio_assembler")
  *
  * val speechToText: HubertForCTC = HubertForCTC
  *   .pretrained()
  *   .setInputCols("audio_assembler")
  *   .setOutputCol("text")
  *
  * val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))
  *
  * val bufferedSource =
  *   scala.io.Source.fromFile("src/test/resources/audio/csv/audio_floats.csv")
  *
  * val rawFloats = bufferedSource
  *   .getLines()
  *   .map(_.split(",").head.trim.toFloat)
  *   .toArray
  * bufferedSource.close
  *
  * val processedAudioFloats = Seq(rawFloats).toDF("audio_content")
  *
  * val result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
  * result.select("text.result").show(truncate = false)
  * +------------------------------------------------------------------------------------------+
  * |result                                                                                    |
  * +------------------------------------------------------------------------------------------+
  * |[MISTER QUILTER IS THE APOSTLE OF THE MIDLE CLASES AND WE ARE GLAD TO WELCOME HIS GOSPEL ]|
  * +------------------------------------------------------------------------------------------+
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
class HubertForCTC(override val uid: String) extends Wav2Vec2ForCTC(uid) {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("HubertForCTC"))

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)

    getEngine match {

      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          "_hubert_ctc",
          HubertForCTC.tfFile,
          configProtoBytes = getConfigProtoBytes)

      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          "_hubert_ctc",
          HubertForCTC.onnxFile)
    }
  }
}

trait ReadablePretrainedHubertForAudioModel
    extends ParamsAndFeaturesReadable[HubertForCTC]
    with HasPretrained[HubertForCTC] {
  override val defaultModelName: Some[String] = Some("asr_hubert_large_ls960")

  /** Java compliant-overrides */
  override def pretrained(): HubertForCTC = super.pretrained()

  override def pretrained(name: String): HubertForCTC = super.pretrained(name)

  override def pretrained(name: String, lang: String): HubertForCTC =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): HubertForCTC =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadHubertForAudioDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[HubertForCTC] =>

  override val tfFile: String = "hubert_ctc_tensorflow"
  override val onnxFile: String = "hubert_ctc_onnx"

  def readTensorflow(instance: HubertForCTC, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tf = readTensorflowModel(path, spark, "_hubert_ctc_tf", initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tf), None)
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_hubert_ctc_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper))
    }
  }

  addReader(readTensorflow)

  def loadSavedModel(modelPath: String, spark: SparkSession): HubertForCTC = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabJsonContent = loadJsonStringAsset(localModelPath, "vocab.json")
    val vocabJsonMap =
      parse(vocabJsonContent, useBigIntForLong = true).values
        .asInstanceOf[Map[String, BigInt]]

    val preprocessorConfigJsonContent =
      loadJsonStringAsset(localModelPath, "preprocessor_config.json")
    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonContent)

    /*Universal parameters for all engines*/
    val annotatorModel = new HubertForCTC()
      .setVocabulary(vocabJsonMap)
      .setDoNormalize(preprocessorConfig.do_normalize)
      .setFeatureSize(preprocessorConfig.feature_size)
      .setPaddingSide(preprocessorConfig.padding_side)
      .setPaddingValue(preprocessorConfig.padding_value)
      .setReturnAttentionMask(preprocessorConfig.return_attention_mask)
      .setSamplingRate(preprocessorConfig.sampling_rate)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) =
          TensorflowWrapper.read(localModelPath, zipped = false, useBundle = true)

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, Some(wrapper), None)
      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[HubertForCTC]]. Please refer to that class for the
  * documentation.
  */
object HubertForCTC extends ReadablePretrainedHubertForAudioModel with ReadHubertForAudioDLModel
