package com.johnsnowlabs.ml.tensorflow.sign

import org.slf4j.{Logger, LoggerFactory}
import org.tensorflow.SavedModelBundle
import org.tensorflow.proto.framework.TensorInfo

import java.util
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex
import scala.util.{Failure, Success, Try}

object BertTFSignManager {

  private[BertTFSignManager] val logger: Logger = LoggerFactory.getLogger("BertTFSignManager")

  def apply(tfSignatureType: String = "JSL",
            tokenIdsValue: String = BertTFSignConstants.TokenIds.value,
            maskIdsValue: String = BertTFSignConstants.MaskIds.value,
            segmentIdsValue: String = BertTFSignConstants.SegmentIds.value,
            embeddingsValue: String = BertTFSignConstants.Embeddings.value,
            sentenceEmbeddingsValue: String = BertTFSignConstants.SentenceEmbeddings.value) =

    tfSignatureType.toUpperCase match {
      case "JSL" =>
        Map[String, String](
          BertTFSignConstants.TokenIds.key -> tokenIdsValue,
          BertTFSignConstants.MaskIds.key -> maskIdsValue,
          BertTFSignConstants.SegmentIds.key -> segmentIdsValue,
          BertTFSignConstants.Embeddings.key -> embeddingsValue,
          BertTFSignConstants.SentenceEmbeddings.key -> sentenceEmbeddingsValue)
      case _ => throw new Exception("Model provider not available.")
    }

  def getBertTokenIdsKey(): String = {BertTFSignConstants.TokenIds.key}
  def getBertTokenIdsValue(): String = {BertTFSignConstants.TokenIds.value}

  def getBertMaskIdsKey(): String = {BertTFSignConstants.TokenIds.key}
  def getBertMaskIdsValue(): String = {BertTFSignConstants.TokenIds.value}

  def getBertSegmentIdsKey(): String = {BertTFSignConstants.TokenIds.key}
  def getBertSegmentIdsValue(): String = {BertTFSignConstants.TokenIds.value}

  def getBertEmbeddingsKey(): String = {BertTFSignConstants.TokenIds.key}
  def getBertEmbeddingsValue(): String = {BertTFSignConstants.TokenIds.value}

  def getBertSentenceEmbeddingsKey(): String = {BertTFSignConstants.TokenIds.key}
  def getBertSentenceEmbeddingsValue(): String = {BertTFSignConstants.TokenIds.value}

  /** Return a formatted map of key -> value for model signature objects */
  def convertToAdoptedKeys(matched: List[((String, String, String), List[String])]) = {
    matched.map(e => BertTFSignConstants.toAdoptedKeys(e._1._2) -> e._1._3).toMap
  }

  /** Extract signatures from actual model
   * @param model: a SavedModelBundle object
   * @return a list of tuples of type (OperationType, key, TFInfoName)
   * */
  def getSignaturesFromModel(model: SavedModelBundle) = {
    import collection.JavaConverters._

    val InputOp = "feed"
    val OutputOp = "fetch"

    val modelSignatures = new ListBuffer[(String, String, String)]()

    if (model.metaGraphDef.hasGraphDef && model.metaGraphDef.getSignatureDefCount > 0) {
      for (sigDef <- model.metaGraphDef.getSignatureDefMap.values.asScala) {
        val inputs: util.Map[String, TensorInfo] = sigDef.getInputsMap
        for (e <- inputs.entrySet.asScala) {
          val key: String = e.getKey
          val tfInfo: TensorInfo = e.getValue
          logger.debug("\nSignatureDef InputMap key: " + key + " tfInfo: " + tfInfo.getName)
          modelSignatures += ((InputOp, key, tfInfo.getName))
        }
      }
      for (sigDef <- model.metaGraphDef.getSignatureDefMap.values.asScala) {
        val outputs: util.Map[String, TensorInfo] = sigDef.getOutputsMap
        for (e <- outputs.entrySet.asScala) {
          val key: String = e.getKey
          val tfInfo: TensorInfo = e.getValue
          logger.debug("\nSignatureDef OutputMap key: " + key + " tfInfo: " + tfInfo.getName)
          modelSignatures += ((OutputOp, key, tfInfo.getName))
        }
      }
    }
    modelSignatures.toList
  }

  /**
   * Extract input and output signatures from TF saved models
   *
   * @param tags tags to load from model
   * @param modelProvider model framework provider, i.e. default, TF2 or HF
   * @param savedModelDir saved model path
   * @return the list ot matching signatures as tuples
   * */
  def extractSignatures(tags: Array[String] = Array("serve"),
                        modelProvider: String = "JSL",
                        savedModelDir: String) = {

    val model =
      Try(SavedModelBundle.load(savedModelDir, tags: _*)) match {
        case Success(bundle) => bundle
        case Failure(s) => throw new Exception(s"Could not retrieve the SavedModelBundle + ${s.printStackTrace()}")
      }

    val signatureCandidates = getSignaturesFromModel(model)
    logger.debug(signatureCandidates.toString)

    /** Regex matcher */
    def findTFKeyMatch(candidate: String, key: Regex) = {
      val pattern = key
      val res = pattern.unapplySeq(candidate)
      res.isDefined
    }

    /**
     * Extract matches from candidate key and model signatures
     *
     * @param candidate: the candidate key name
     * @param modelProvider: the model provider in between default, TF2 and HF to select the proper keys
     * @return a list of matching keys as strings
     * */
    def extractCandidateMatches(candidate: String, modelProvider: String) : List[String] = {
      val ReferenceKeys: Array[Regex] = BertTFSignConstants.getSignaturePatterns(modelProvider)

      val matches = (
        for (refKey <- ReferenceKeys if findTFKeyMatch(candidate, refKey)) yield {
          refKey
        }).toList
      if (matches.isEmpty) List("N/A") else matches.mkString(",").split(",").toList
    }

    val matched = signatureCandidates.map(s => (s, extractCandidateMatches(s._2, modelProvider)))

    Option(convertToAdoptedKeys(matched))
  }
}
