package com.johnsnowlabs.ml.tensorflow.sign


object TFSignConstants {

  sealed trait TFInfoName

  case object TokenIds extends TFInfoName {
    val key: String = "ids"
    val value: String = "input_ids:0"
  }

  case object MaskIds extends TFInfoName {
    val key: String = "mask"
    val value: String = "input_mask:0"
  }

  case object SegmentIds extends TFInfoName {
    val key: String = "segs"
    val value: String = "segment_ids:0"
  }

  case object Embeddings extends TFInfoName {
    val key: String = "out"
    val value: String = "sequence_output:0"
  }

  case object SentenceEmbeddings extends TFInfoName {
    val key: String = "s_out"
    val value: String = "pooled_output:0"
  }
}
