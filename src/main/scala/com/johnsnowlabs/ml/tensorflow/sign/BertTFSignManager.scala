package com.johnsnowlabs.ml.tensorflow.sign

object BertTFSignManager {

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
}
