/*
 * Copyright 2017-2023 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.ml.ai.{RoBertaClassification, ZeroShotNerClassification}
import com.johnsnowlabs.ml.onnx.OnnxWrapper
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowWrapper}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, TOKEN}
import com.johnsnowlabs.nlp.annotator.RoBertaForQuestionAnswering
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, HasPretrained, ParamsAndFeaturesReadable}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.{FloatParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import java.util
import scala.collection.JavaConverters._

/** ZeroShotNerModel implements zero shot named entity recognition by utilizing RoBERTa
  * transformer models fine tuned on a question answering task.
  *
  * Its input is a list of document annotations and it automatically generates questions which are
  * used to recognize entities. The definitions of entities is given by a dictionary structures,
  * specifying a set of questions for each entity. The model is based on
  * RoBertaForQuestionAnswering.
  *
  * For more extended examples see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/named-entity-recognition/ZeroShot_NER.ipynb Examples]]
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val zeroShotNer = ZeroShotNerModel.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("zer_shot_ner")
  * }}}
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Zero-Shot-NER Models Hub]].
  *
  * ==Example==
  * {{{
  *  val documentAssembler = new DocumentAssembler()
  *    .setInputCol("text")
  *    .setOutputCol("document")
  *
  *  val sentenceDetector = new SentenceDetector()
  *    .setInputCols(Array("document"))
  *    .setOutputCol("sentences")
  *
  *  val zeroShotNer = ZeroShotNerModel
  *    .pretrained()
  *    .setEntityDefinitions(
  *      Map(
  *        "NAME" -> Array("What is his name?", "What is her name?"),
  *        "CITY" -> Array("Which city?")))
  *    .setPredictionThreshold(0.01f)
  *    .setInputCols("sentences")
  *    .setOutputCol("zero_shot_ner")
  *
  *  val pipeline = new Pipeline()
  *    .setStages(Array(
  *      documentAssembler,
  *      sentenceDetector,
  *      zeroShotNer))
  *
  *  val model = pipeline.fit(Seq("").toDS.toDF("text"))
  *  val results = model.transform(
  *    Seq("Clara often travels between New York and Paris.").toDS.toDF("text"))
  *
  *  results
  *    .selectExpr("document", "explode(zero_shot_ner) AS entity")
  *    .select(
  *      col("entity.result"),
  *      col("entity.metadata.word"),
  *      col("entity.metadata.sentence"),
  *      col("entity.begin"),
  *      col("entity.end"),
  *      col("entity.metadata.confidence"),
  *      col("entity.metadata.question"))
  *    .show(truncate=false)
  *
  * +------+-----+--------+-----+---+----------+------------------+
  * |result|word |sentence|begin|end|confidence|question          |
  * +------+-----+--------+-----+---+----------+------------------+
  * |B-CITY|Paris|0       |41   |45 |0.78655756|Which is the city?|
  * |B-CITY|New  |0       |28   |30 |0.29346612|Which city?       |
  * |I-CITY|York |0       |32   |35 |0.29346612|Which city?       |
  * +------+-----+--------+-----+---+----------+------------------+
  *
  * }}}
  *
  * @see
  *   [[https://arxiv.org/abs/1907.11692]] for details about the RoBERTa transformer
  * @see
  *   [[RoBertaForQuestionAnswering]] for the SparkNLP implementation of RoBERTa question
  *   answering
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
class ZeroShotNerModel(override val uid: String) extends RoBertaForQuestionAnswering {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("ZeroShotNerModel"))

  /** Input Annotator Types: DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  /** Output Annotator Types: NAMED_ENTITY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = NAMED_ENTITY

  /** List of definitions of named entities
    *
    * @group param
    */
  private val entityDefinitions = new MapFeature[String, Array[String]](this, "entityDefinitions")

  /** Set definitions of named entities
    *
    * @group setParam
    */
  def setEntityDefinitions(definitions: Map[String, Array[String]]): this.type = {
    set(this.entityDefinitions, definitions)
  }

  /** Set definitions of named entities
    *
    * @group setParam
    */
  def setEntityDefinitions(definitions: util.HashMap[String, util.List[String]]): this.type = {
    val c = definitions.asScala.mapValues(_.asScala.toList.toArray).toMap
    set(this.entityDefinitions, c)
  }

  /** Get definitions of named entities
    *
    * @group getParam
    */
  private def getEntityDefinitions: scala.collection.immutable.Map[String, Array[String]] = {
    if (!entityDefinitions.isSet)
      return Map.empty
    $$(entityDefinitions)
  }

  def getEntityDefinitionsStr: Array[String] = {
    getEntityDefinitions.map(x => x._1 + "@@@" + x._2.mkString("@@@")).toArray
  }

  var predictionThreshold =
    new FloatParam(this, "predictionThreshold", "Minimal score of predicted entity")

  var ignoreEntities = new StringArrayParam(this, "ignoreEntities", "List of entities to ignore")

  /** Get the minimum entity prediction score
    *
    * @group getParam
    */
  def getPredictionThreshold: Float = $(predictionThreshold)

  /** Set the minimum entity prediction score
    *
    * @group setParam
    */
  def setPredictionThreshold(value: Float): this.type = set(this.predictionThreshold, value)

  /** Get the list of questions to catch the distractor entity
    *
    * @group getParam
    */
  def getIgnoreEntities: Array[String] = $(ignoreEntities)

  /** Get the list of entities which are recognized
    *
    * @group getParam
    */

  def getEntities: Array[String] = getEntityDefinitions.keys.toArray

  /** Set the list of questions to catch the distractor entity
    *
    * @group setParam
    */
  def setIgnoreEntities(value: Array[String]): this.type = set(this.ignoreEntities, value)

  private def getNerQuestionAnnotations
      : scala.collection.immutable.Map[String, Array[Annotation]] = {
    getEntityDefinitions.map(nerDef => {
      (
        nerDef._1,
        nerDef._2.map(nerQ =>
          new Annotation(
            AnnotatorType.DOCUMENT,
            0,
            nerQ.length,
            nerQ,
            Map("entity" -> nerDef._1) ++ Map("ner_question" -> nerQ))))
    })
  }

  private var _model: Option[Broadcast[ZeroShotNerClassification]] = None

  override def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper]): ZeroShotNerModel = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new ZeroShotNerClassification(
            tensorflowWrapper,
            onnxWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            padTokenId,
            false,
            configProtoBytes = getConfigProtoBytes,
            tags = Map.empty[String, Int],
            signatures = getSignatures,
            $$(merges),
            $$(vocabulary))))
    }

    this
  }

  override def getModelIfNotSet: RoBertaClassification = _model.get.value

  setDefault(ignoreEntities -> Array(), predictionThreshold -> 0.01f)

  val maskSymbol = "_"

  private def spansOverlap(span1: (Int, Int), span2: (Int, Int)): Boolean = {
    !((span2._1 > span1._2) || (span1._1 > span2._2))
  }

  private def recognizeEntities(
      document: Annotation,
      nerDefs: Map[String, Array[Annotation]]): Seq[Annotation] = {
    val docPredictions = nerDefs
      .flatMap(nerDef => {
        val nerBatch = nerDef._2.map(nerQuestion => Array(nerQuestion, document)).toSeq
        val entityPredictions = super
          .batchAnnotate(nerBatch)
          .zip(nerBatch.map(_.head.result))
          .map(x => (x._1.head, x._2))
          .filter(x => x._1.result.nonEmpty)
          .filter(x =>
            (if (x._1.metadata.contains("score"))
               x._1.metadata("score").toFloat
             else Float.MinValue) > getPredictionThreshold)
        entityPredictions.map(prediction =>
          new Annotation(
            AnnotatorType.CHUNK,
            prediction._1.begin,
            prediction._1.end,
            prediction._1.result,
            Map(
              "entity" -> nerDef._1,
              "sentence" -> document.metadata("sentence"),
              "word" -> prediction._1.result,
              "confidence" -> prediction._1.metadata("score"),
              "question" -> prediction._2)))
      })
      .toSeq
    // Detect overlapping predictions and choose the one with the higher score
    docPredictions
      .filter(x => ! $(ignoreEntities).contains(x.metadata("entity"))) // Discard ignored entities
      .filter(prediction => {
        !docPredictions
          .filter(_ != prediction)
          .exists(otherPrediction => {
            spansOverlap(
              (prediction.begin, prediction.end),
              (otherPrediction.begin, otherPrediction.end)) && (otherPrediction.metadata(
              "confidence") > prediction.metadata("confidence"))
          })
      })

  }

  def maskEntity(document: Annotation, entity: Annotation): String = {
    val entityStart = entity.begin - document.begin
    val entityEnd = entity.end - document.begin
    //    println(document.result.slice(0, entityStart) + maskSymbol + {entityStart to entityEnd - 2}.map(_ => " ").mkString + document.result.slice(entityEnd, $(maxSentenceLength)))
    document.result.slice(0, entityStart) + maskSymbol + {
      entityStart to entityEnd - 2
    }.map(_ => " ").mkString + document.result.slice(entityEnd, $(maxSentenceLength))
  }

  def recognizeMultipleEntities(
      document: Annotation,
      nerDefs: Map[String, Array[Annotation]],
      recognizedEntities: Seq[Annotation] = Seq()): Seq[Annotation] = {
    val newEntities = recognizeEntities(document, nerDefs)
      .filter(entity =>
        (!recognizedEntities
          .exists(recognizedEntity =>
            spansOverlap(
              (entity.begin, entity.end),
              (recognizedEntity.begin, recognizedEntity.end)))) && (entity.result != maskSymbol))

    newEntities ++ newEntities.flatMap { entity =>
      val newDoc = new Annotation(
        document.annotatorType,
        document.begin,
        document.end,
        maskEntity(document, entity),
        document.metadata)
      recognizeMultipleEntities(
        newDoc,
        nerDefs.filter(x => x._1 == entity.metadata("entity")),
        recognizedEntities ++ newEntities)
    }
  }

  def isTokenInEntity(token: Annotation, entity: Annotation): Boolean = {
    (
      token.metadata("sentence") == entity.metadata("sentence")
      && (token.begin >= entity.begin) && (token.end <= entity.end)
    )
  }

  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {

    batchedAnnotations.map(annotations => {
      val documents = annotations
        .filter(_.annotatorType == AnnotatorType.DOCUMENT)
        .toSeq
      val tokens = annotations.filter(_.annotatorType == AnnotatorType.TOKEN)
      val entities = documents.flatMap { doc =>
        recognizeMultipleEntities(doc, getNerQuestionAnnotations).flatMap { entity =>
          tokens
            .filter(t => isTokenInEntity(t, entity))
            .zipWithIndex
            .map { case (token, i) =>
              val bioPrefix = if (i == 0) "B-" else "I-"
              new Annotation(
                annotatorType = AnnotatorType.NAMED_ENTITY,
                begin = token.begin,
                end = token.end,
                result = bioPrefix + entity.metadata("entity"),
                metadata = Map(
                  "sentence" -> entity.metadata("sentence"),
                  "word" -> token.result,
                  "confidence" -> entity.metadata("confidence"),
                  "question" -> entity.metadata("question")))
            }
        }.toList
      }
      tokens
        .map(token => {
          val entity = entities.find(e => isTokenInEntity(token, e))
          if (entity.nonEmpty) {
            entity.get
          } else {
            new Annotation(
              annotatorType = AnnotatorType.NAMED_ENTITY,
              begin = token.begin,
              end = token.end,
              result = "O",
              metadata = Map("sentence" -> token.metadata("sentence"), "word" -> token.result))
          }
        })
        .toSeq
    })
  }

}

trait ReadablePretrainedZeroShotNer
    extends ParamsAndFeaturesReadable[ZeroShotNerModel]
    with HasPretrained[ZeroShotNerModel] {
  override val defaultModelName: Some[String] = Some("zero_shot_ner_roberta")

  /** Java compliant-overrides */
  override def pretrained(): ZeroShotNerModel =
    pretrained(defaultModelName.get, defaultLang, defaultLoc)

  override def pretrained(name: String): ZeroShotNerModel =
    pretrained(name, defaultLang, defaultLoc)

  override def pretrained(name: String, lang: String): ZeroShotNerModel =
    pretrained(name, lang, defaultLoc)

  override def pretrained(name: String, lang: String, remoteLoc: String): ZeroShotNerModel = {
    try {
      ZeroShotNerModel.getFromRoBertaForQuestionAnswering(
        ResourceDownloader
          .downloadModel(RoBertaForQuestionAnswering, name, Option(lang), remoteLoc))
    } catch {
      case _: java.lang.RuntimeException =>
        ResourceDownloader.downloadModel(ZeroShotNerModel, name, Option(lang), remoteLoc)
    }
  }

  override def load(path: String): ZeroShotNerModel = {
    try {
      super.load(path)
    } catch {
      case e: java.lang.ClassCastException =>
        try {
          ZeroShotNerModel.getFromRoBertaForQuestionAnswering(
            RoBertaForQuestionAnswering.load(path))
        } catch {
          case _: Throwable => throw e
        }
    }
  }
}

trait ReadZeroShotNerDLModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[ZeroShotNerModel] =>

  override val tfFile: String = "roberta_classification_tensorflow"

  def readTensorflow(instance: ZeroShotNerModel, path: String, spark: SparkSession): Unit = {

    val tfWrapper =
      readTensorflowModel(path, spark, "_roberta_classification_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, Some(tfWrapper), None)
  }

  addReader(readTensorflow)
}
object ZeroShotNerModel extends ReadablePretrainedZeroShotNer with ReadZeroShotNerDLModel {

  def apply(model: PipelineStage): PipelineStage = {
    model match {
      case answering: RoBertaForQuestionAnswering if !model.isInstanceOf[ZeroShotNerModel] =>
        getFromRoBertaForQuestionAnswering(answering)
      case _ =>
        model
    }
  }

  def getFromRoBertaForQuestionAnswering(model: RoBertaForQuestionAnswering): ZeroShotNerModel = {
    val spark = SparkSession.builder.getOrCreate()

    val newModel = new ZeroShotNerModel()
      .setVocabulary(
        model.vocabulary.get.getOrElse(throw new RuntimeException("Vocabulary not set")))
      .setMerges(model.merges.get.getOrElse(throw new RuntimeException("Merges not set")))
      .setCaseSensitive(model.getCaseSensitive)
      .setBatchSize(model.getBatchSize)

    if (model.signatures.isSet)
      newModel.setSignatures(
        model.signatures.get.getOrElse(throw new RuntimeException("Signatures not set")))

    newModel.setModelIfNotSet(spark, model.getModelIfNotSet.tensorflowWrapper, None)

    model
      .extractParamMap()
      .toSeq
      .foreach(x => {
        newModel.set(x.param.name, x.value)
      })

    newModel
  }
}
