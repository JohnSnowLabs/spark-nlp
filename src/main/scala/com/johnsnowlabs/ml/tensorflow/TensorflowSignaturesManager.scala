package com.johnsnowlabs.ml.tensorflow


object TensorflowSignaturesManager{

  def apply(tfSignatureType: String = "JSL",
            tokenIdsKey: String = "input_ids:0",
            maskIdsKey: String = "input_mask:0",
            segmentIdsKey: String = "segment_ids:0",
            embeddingsKey: String = "sequence_output:0",
            sentenceEmbeddingsKey: String = "pooled_output:0") =

    tfSignatureType.toUpperCase match {
      case "JSL" =>
        Map[String, String](
          "ids" -> tokenIdsKey,
          "mask" -> maskIdsKey,
          "segs" -> segmentIdsKey,
          "out" -> embeddingsKey,
          "s_out" -> sentenceEmbeddingsKey)
      case _ => throw new Exception("Model provider not available.")
    }
}