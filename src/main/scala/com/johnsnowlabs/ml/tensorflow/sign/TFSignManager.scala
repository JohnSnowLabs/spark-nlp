package com.johnsnowlabs.ml.tensorflow.sign

object TFSignManager {

  def apply(tfSignatureType: String = "JSL",
            tokenIdsValue: String = TFSignConstants.TokenIds.value,
            maskIdsValue: String = TFSignConstants.MaskIds.value,
            segmentIdsValue: String = TFSignConstants.SegmentIds.value,
            embeddingsValue: String = TFSignConstants.Embeddings.value,
            sentenceEmbeddingsValue: String = TFSignConstants.SentenceEmbeddings.value) =

    tfSignatureType.toUpperCase match {
      case "JSL" =>
        Map[String, String](
          TFSignConstants.TokenIds.key -> tokenIdsValue,
          TFSignConstants.MaskIds.key -> maskIdsValue,
          TFSignConstants.SegmentIds.key -> segmentIdsValue,
          TFSignConstants.Embeddings.key -> embeddingsValue,
          TFSignConstants.SentenceEmbeddings.key -> sentenceEmbeddingsValue)
      case _ => throw new Exception("Model provider not available.")
    }
}
