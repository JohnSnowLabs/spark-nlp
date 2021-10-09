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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow.wrap.TFWrapper
import com.johnsnowlabs.ml.tensorflow.{TensorflowWrapper, _}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.BpeTokenizer
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import java.io.File

/**
 * RoBertaForTokenClassification can load RoBERTa Models with a token classification head on top (a linear layer on top of the hidden-states output)
 * e.g. for Named-Entity-Recognition (NER) tasks.
 *
 * Pretrained models can be loaded with `pretrained` of the companion object:
 * {{{
 * val tokenClassifier = RoBertaForTokenClassification.pretrained()
 *   .setInputCols("token", "document")
 *   .setOutputCol("label")
 * }}}
 * The default model is `"roberta_base_token_classifier_conll03"`, if no name is provided.
 *
 * For available pretrained models please see the [[https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition Models Hub]].
 *
 * and the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/RoBertaForTokenClassificationTestSpec.scala RoBertaForTokenClassificationTestSpec]].
 * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. The Spark NLP Workshop
 * example shows how to import them [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
 *
 * ==Example==
 * {{{
 * import spark.implicits._
 * import com.johnsnowlabs.nlp.base._
 * import com.johnsnowlabs.nlp.annotator._
 * import org.apache.spark.ml.Pipeline
 *
 * val documentAssembler = new DocumentAssembler()
 *   .setInputCol("text")
 *   .setOutputCol("document")
 *
 * val tokenizer = new Tokenizer()
 *   .setInputCols("document")
 *   .setOutputCol("token")
 *
 * val tokenClassifier = RoBertaForTokenClassification.pretrained()
 *   .setInputCols("token", "document")
 *   .setOutputCol("label")
 *   .setCaseSensitive(true)
 *
 * val pipeline = new Pipeline().setStages(Array(
 *   documentAssembler,
 *   tokenizer,
 *   tokenClassifier
 * ))
 *
 * val data = Seq("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London").toDF("text")
 * val result = pipeline.fit(data).transform(data)
 *
 * result.select("label.result").show(false)
 * +------------------------------------------------------------------------------------+
 * |result                                                                              |
 * +------------------------------------------------------------------------------------+
 * |[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
 * +------------------------------------------------------------------------------------+
 * }}}
 *
 * @see [[RoBertaForTokenClassification]] for sentence-level embeddings
 * @see [[https://nlp.johnsnowlabs.com/docs/en/annotators Annotators Main Page]] for a list of transformer based classifiers
 * @param uid required uid for storing annotator to disk
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
class RoBertaForTokenClassification(override val uid: String)
  extends AnnotatorModel[RoBertaForTokenClassification]
    with HasBatchedAnnotate[RoBertaForTokenClassification]
    with WriteTensorflowModel
    with HasCaseSensitiveProperties {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  def this() = this(Identifiable.randomUID("RoBertaForTokenClassification"))


  /**
   * Input Annotator Types: DOCUMENT, TOKEN
   *
   * @group anno
   */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)

  /**
   * Output Annotator Types: WORD_EMBEDDINGS
   *
   * @group anno
   */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.NAMED_ENTITY

  def sentenceStartTokenId: Int = {
    $$(vocabulary)("<s>")
  }

  def sentenceEndTokenId: Int = {
    $$(vocabulary)("</s>")
  }

  def padTokenId: Int = {
    $$(vocabulary)("<pad>")
  }

  /**
   * Vocabulary used to encode the words to ids with WordPieceEncoder
   *
   * @group param
   * */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)


  /**
   * Labels used to decode predicted IDs back to string tags
   *
   * @group param
   * */
  val labels: MapFeature[String, Int] = new MapFeature(this, "labels")

  /** @group setParam */
  def setLabels(value: Map[String, Int]): this.type = set(labels, value)

  /** @group getParam */
  def getLabels: Map[String, Int] = $$(labels)


  /**
   * Holding merges.txt coming from RoBERTa model
   *
   * @group param
   */
  val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges")

  /** @group setParam */
  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

  /** ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
   *
   * @group param
   * */
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): RoBertaForTokenClassification.this.type = set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process (Default: `128`)
   *
   * @group param
   * */
  val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(value <= 512, "RoBERTa models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /**
   * It contains TF model signatures for the laded saved model
   *
   * @group param
   * */
  val signatures = new MapFeature[String, String](model = this, name = "signatures")

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  private var _model: Option[Broadcast[TensorflowRoBertaClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflowWrapper: TFWrapper[_]): RoBertaForTokenClassification = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowRoBertaClassification(
            tensorflowWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            padTokenId,
            configProtoBytes = getConfigProtoBytes,
            tags = getLabels,
            signatures = getSignatures
          )
        )
      )
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: TensorflowRoBertaClassification = _model.get.value


  /** Whether to lowercase tokens or not
   *
   * @group setParam
   * */
  override def setCaseSensitive(value: Boolean): this.type = {
    if (get(caseSensitive).isEmpty)
      set(this.caseSensitive, value)
    this
  }

  setDefault(
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> true
  )

  def tokenizeWithAlignment(tokens: Seq[TokenizedSentence]): Seq[WordpieceTokenizedSentence] = {
    val bpeTokenizer = BpeTokenizer.forModel(
      "roberta",
      merges = $$(merges),
      vocab = $$(vocabulary),
      padWithSentenceTokens = false
    )

    tokens.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens = tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map { token =>
        val content = if ($(caseSensitive)) token.token else token.token.toLowerCase()
        val sentenceBegin = token.begin
        val sentenceEnd = token.end
        val sentenceInedx = tokenIndex.sentenceIndex
        val result = bpeTokenizer.tokenize(Sentence(content, sentenceBegin, sentenceEnd, sentenceInedx))
        if (result.nonEmpty) result.head else IndexedToken("")
      }
      val wordpieceTokens = bertTokens.flatMap(token => bpeTokenizer.encode(token)).take($(maxSentenceLength))
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  /**
   * takes a document and annotations and produces new annotations of this annotator's annotation type
   *
   * @param batchedAnnotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
   * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
   */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    val batchedTokenizedSentences: Array[Array[TokenizedSentence]] = batchedAnnotations.map(annotations =>
      TokenizedWithSentence.unpack(annotations).toArray
    ).toArray
    /*Return empty if the real tokens are empty*/
    if (batchedTokenizedSentences.nonEmpty) batchedTokenizedSentences.map(tokenizedSentences => {
      val tokenized = tokenizeWithAlignment(tokenizedSentences)

      getModelIfNotSet.predict(
        tokenized,
        tokenizedSentences,
        $(batchSize),
        $(maxSentenceLength)
      )
    }) else {
      Seq(Seq.empty[Annotation])
    }
  }


  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(path, spark, getModelIfNotSet.tensorflowWrapper, "_roberta_classification", RoBertaForTokenClassification.tfFile, configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedRoBertaForTokenModel extends ParamsAndFeaturesReadable[RoBertaForTokenClassification] with HasPretrained[RoBertaForTokenClassification] {
  override val defaultModelName: Some[String] = Some("roberta_base_token_classifier_conll03")

  /** Java compliant-overrides */
  override def pretrained(): RoBertaForTokenClassification = super.pretrained()

  override def pretrained(name: String): RoBertaForTokenClassification = super.pretrained(name)

  override def pretrained(name: String, lang: String): RoBertaForTokenClassification = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): RoBertaForTokenClassification = super.pretrained(name, lang, remoteLoc)
}

trait ReadRoBertaForTokenTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[RoBertaForTokenClassification] =>

  override val tfFile: String = "roberta_classification_tensorflow"

  def readTensorflow(instance: RoBertaForTokenClassification, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_roberta_classification_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(tfModelPath: String, spark: SparkSession): RoBertaForTokenClassification = {

    val f = new File(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")

    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $tfModelPath"
    )

    val vocabFile = new File(tfModelPath + "/assets", "vocab.txt")
    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(vocabFile.exists(), s"Vocabulary file vocab.txt not found in folder $tfModelPath")

    val mergesFile = new File(tfModelPath + "/assets", "merges.txt")
    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(mergesFile.exists(), s"merges file merges.txt not found in folder $tfModelPath")

    val vocabResource = new ExternalResource(vocabFile.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

    val mergesResource = new ExternalResource(mergesFile.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val merges = ResourceHelper.parseLines(mergesResource)

    val bytePairs: Map[(String, String), Int] = merges.map(_.split(" "))
      .filter(w => w.length == 2)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex.toMap

    val labelsPath = new File(tfModelPath + "/assets", "labels.txt")
    require(labelsPath.exists(), s"Labels file labels.txt not found in folder $tfModelPath/assets/")

    val labelsResource = new ExternalResource(labelsPath.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val labels = ResourceHelper.parseLines(labelsResource).zipWithIndex.toMap

    val (wrapper, signatures) = TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    /** the order of setSignatures is important if we use getSignatures inside setModelIfNotSet */
    new RoBertaForTokenClassification()
      .setVocabulary(words)
      .setMerges(bytePairs)
      .setLabels(labels)
      .setSignatures(_signatures)
      .setModelIfNotSet(spark, wrapper)
  }
}

/**
 * This is the companion object of [[RoBertaForTokenClassification]]. Please refer to that class for the documentation.
 */
object RoBertaForTokenClassification extends ReadablePretrainedRoBertaForTokenModel with ReadRoBertaForTokenTensorflowModel
