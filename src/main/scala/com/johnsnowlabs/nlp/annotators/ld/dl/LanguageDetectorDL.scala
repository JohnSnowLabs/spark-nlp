/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.ld.dl

import com.johnsnowlabs.ml.tensorflow.wrap.TFWrapper

import java.io.File
import com.johnsnowlabs.ml.tensorflow.{TensorflowWrapper, _}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import scala.collection.immutable.ListMap

/**
 * Language Identification and Detection by using CNN and RNN architectures in TensorFlow.
 *
 * `LanguageDetectorDL` is an annotator that detects the language of documents or sentences depending on the inputCols.
 * The models are trained on large datasets such as Wikipedia and Tatoeba.
 * Depending on the language (how similar the characters are), the LanguageDetectorDL works
 * best with text longer than 140 characters.
 * The output is a language code in [[https://en.wikipedia.org/wiki/List_of_Wikipedias Wiki Code style]].
 *
 * Pretrained models can be loaded with `pretrained` of the companion object:
 * {{{
 * Val languageDetector = LanguageDetectorDL.pretrained()
 *   .setInputCols("sentence")
 *   .setOutputCol("language")
 * }}}
 * The default model is `"ld_wiki_tatoeba_cnn_21"`, default language is `"xx"` (meaning multi-lingual),
 * if no values are provided.
 * For available pretrained models please see the [[https://nlp.johnsnowlabs.com/models?task=Language+Detection Models Hub]].
 *
 * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb Spark NLP Workshop]]
 * And the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ld/dl/LanguageDetectorDLTestSpec.scala LanguageDetectorDLTestSpec]].
 *
 * ==Example==
 * {{{
 * import spark.implicits._
 * import com.johnsnowlabs.nlp.base.DocumentAssembler
 * import com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL
 * import org.apache.spark.ml.Pipeline
 *
 * val documentAssembler = new DocumentAssembler()
 *   .setInputCol("text")
 *   .setOutputCol("document")
 *
 * val languageDetector = LanguageDetectorDL.pretrained()
 *   .setInputCols("document")
 *   .setOutputCol("language")
 *
 * val pipeline = new Pipeline()
 *   .setStages(Array(
 *     documentAssembler,
 *     languageDetector
 *   ))
 *
 * val data = Seq(
 *   "Spark NLP is an open-source text processing library for advanced natural language processing for the Python, Java and Scala programming languages.",
 *   "Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala.",
 *   "Spark NLP ist eine Open-Source-Textverarbeitungsbibliothek für fortgeschrittene natürliche Sprachverarbeitung für die Programmiersprachen Python, Java und Scala."
 * ).toDF("text")
 * val result = pipeline.fit(data).transform(data)
 *
 * result.select("language.result").show(false)
 * +------+
 * |result|
 * +------+
 * |[en]  |
 * |[fr]  |
 * |[de]  |
 * +------+
 * }}}
 *
 * @groupname anno Annotator types
 * @groupdesc anno Required input and expected output annotator types
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
 * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 * */
class LanguageDetectorDL(override val uid: String)
  extends AnnotatorModel[LanguageDetectorDL]
    with HasSimpleAnnotate[LanguageDetectorDL]
    with WriteTensorflowModel {

  def this() = this(Identifiable.randomUID("LANGUAGE_DETECTOR_DL"))

  /** Alphabet used to feed the TensorFlow model for prediction
   *
   * @group param
   * */
  val alphabet: MapFeature[String, Int] = new MapFeature(this, "alphabet")

  /** Language used to map prediction to ISO 639-1 language codes
   *
   * @group param
   * */
  val language: MapFeature[String, Int] = new MapFeature(this, "language")

  /** The minimum threshold for the final result, otherwise it will be either `"unk"` or the value set in
   * `thresholdLabel` (Default: `0.1f`).
   * Value is between 0.0 to 1.0. Try to set this lower if your text is hard to predict
   *
   * @group param
   * */
  val threshold = new FloatParam(this, "threshold", "The minimum threshold for the final result otherwise it will be either Unknown or the value set in thresholdLabel.")

  /** Value for the classification, if confidence is less than `threshold` (Default: `"unk"`).
   *
   * @group param
   * */
  val thresholdLabel = new Param[String](this, "thresholdLabel", "In case the score is less than threshold, what should be the label. Default is Unknown.")

  /** Output average of sentences instead of one output per sentence (Default: `true`).
   *
   * @group param
   * */
  val coalesceSentences = new BooleanParam(this, "coalesceSentences", "If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to true.")

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
   *
   * @group param
   * */
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** Languages the model was trained with.
   *
   * @group param
   */
  val languages = new StringArrayParam(this, "languages", "keep an internal copy of languages for Python")

  /** @group setParam */
  def setLanguage(value: Map[String, Int]): this.type = {
    if (get(language).isEmpty)
      set(this.language, value)
    this
  }

  /** @group setParam */
  def setAlphabet(value: Map[String, Int]): this.type = {
    if (get(language).isEmpty)
      set(alphabet, value)
    this
  }

  /** @group setParam */
  def setThreshold(threshold: Float): this.type = set(this.threshold, threshold)

  /** @group setParam */
  def setThresholdLabel(label: String): this.type = set(this.thresholdLabel, label)

  /** @group setParam */
  def setCoalesceSentences(value: Boolean): this.type = set(coalesceSentences, value)

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): LanguageDetectorDL.this.type = set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getLanguage: Array[String] = {
    val langs = $$(language).keys.toArray
    set(languages, langs)
    langs
  }

  /** @group getParam */
  def getThreshold: Float = $(this.threshold)

  /** @group getParam */
  def getThresholdLabel: String = $(this.thresholdLabel)

  /** @group getParam */
  def getCoalesceSentences: Boolean = $(coalesceSentences)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  setDefault(
    inputCols -> Array("document"),
    outputCol -> "language",
    threshold -> 0.1f,
    thresholdLabel -> "unk",
    coalesceSentences -> true
  )

  private var _model: Option[Broadcast[TensorflowLD]] = None

  /** @group getParam */
  def getModelIfNotSet: TensorflowLD = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflow: TFWrapper[_]): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowLD(
            tensorflow,
            configProtoBytes = getConfigProtoBytes,
            ListMap($$(language).toSeq.sortBy(_._2): _*),
            ListMap($$(alphabet).toSeq.sortBy(_._2): _*)
          )
        )
      )
    }

    this
  }

  /**
   * Takes a document and annotations and produces new annotations of this annotator's annotation type
   *
   * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
   * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
   */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = SentenceSplit.unpack(annotations)
    val nonEmptySentences = sentences.filter(_.content.nonEmpty)
    if (nonEmptySentences.nonEmpty) {
      getModelIfNotSet.calculateLanguageIdentification(
        nonEmptySentences,
        $(threshold),
        $(thresholdLabel),
        $(coalesceSentences)
      )
    } else {
      Seq.empty[Annotation]
    }
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.LANGUAGE

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(path, spark, getModelIfNotSet.tensorflow, "_ld", LanguageDetectorDL.tfFile, configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedLanguageDetectorDLModel extends ParamsAndFeaturesReadable[LanguageDetectorDL] with HasPretrained[LanguageDetectorDL] {
  override val defaultModelName: Some[String] = Some("ld_wiki_tatoeba_cnn_21")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): LanguageDetectorDL = super.pretrained()

  override def pretrained(name: String): LanguageDetectorDL = super.pretrained(name)

  override def pretrained(name: String, lang: String): LanguageDetectorDL = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): LanguageDetectorDL = super.pretrained(name, lang, remoteLoc)
}

trait ReadLanguageDetectorDLTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[LanguageDetectorDL] =>

  override val tfFile: String = "ld_tensorflow"

  def readTensorflow(instance: LanguageDetectorDL, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_ld_tf")
    instance.setModelIfNotSet(spark, tf)
    // This allows for Python to access getLanguages function
    val t = instance.language.get.toArray
    val r = t(0).keys.toArray
    instance.set(instance.languages, r)
  }

  addReader(readTensorflow)

  def loadSavedModel(folder: String, spark: SparkSession): LanguageDetectorDL = {

    val f = new File(folder)
    val savedModel = new File(folder, "saved_model.pb")
    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )

    val alphabetPath = new File(folder + "/assets", "alphabet.txt")
    val languagePath = new File(folder + "/assets", "language.txt")

    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(alphabetPath.exists(), s"Alphabet file alphabet.txt not found in folder $folder")
    require(languagePath.exists(), s"Language file language.txt not found in folder $folder")

    val alphabetResource = new ExternalResource(alphabetPath.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val alphabets = ResourceHelper.parseLines(alphabetResource).zipWithIndex.toMap

    val languageResource = new ExternalResource(languagePath.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val languages = ResourceHelper.parseLines(languageResource).zipWithIndex.toMap

    val (wrapper, _) = TensorflowWrapper.read(folder, zipped = false, useBundle = true, tags = Array("serve"))

    new LanguageDetectorDL()
      .setAlphabet(alphabets)
      .setLanguage(languages)
      .setModelIfNotSet(spark, wrapper)
  }
}


/**
 * This is the companion object of [[LanguageDetectorDL]]. Please refer to that class for the documentation.
 */
object LanguageDetectorDL extends ReadablePretrainedLanguageDetectorDLModel with ReadLanguageDetectorDLTensorflowModel
