package com.johnsnowlabs.ml.tensorflow.sign


private [sign] object BertTFSignConstants {

  sealed trait TFInfoNameMapper {
    protected val key: String
    protected val value: String
  }

  case object TokenIds extends TFInfoNameMapper {
    override val key: String = "ids"
    override val value: String = "input_ids:0"
  }

  case object MaskIds extends TFInfoNameMapper {
    override val key: String = "mask"
    override val value: String = "input_mask:0"
  }

  case object SegmentIds extends TFInfoNameMapper {
    override val key: String = "segs"
    override val value: String = "segment_ids:0"
  }

  case object Embeddings extends TFInfoNameMapper {
    override val key: String = "out"
    override val value: String = "sequence_output:0"
  }

  case object SentenceEmbeddings extends TFInfoNameMapper {
    override val key: String = "s_out"
    override val value: String = "pooled_output:0"
  }
}
