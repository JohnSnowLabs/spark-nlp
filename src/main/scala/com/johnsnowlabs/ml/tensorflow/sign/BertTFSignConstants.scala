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

  /** Retrieve signature patterns for a given provider
   * @param modelProvider: the provider library that built the model and the signatures
   * @return reference keys array of signature patterns for a given provider
   * */
  def getSignaturePatterns(modelProvider: String) = {
    val referenceKeys = modelProvider match {
      case "JSL" =>
        Array(
          "(input)(.*)(ids)".r,
          "(input)(.*)(mask)".r,
          "(segment)(.*)(ids)".r,
          "(sequence)(.*)(output)".r,
          "(pooled)(.*)(output)".r)

      case "TF2" =>
        Array(
          "(input_word)(.*)(ids)".r,
          "(input)(.*)(mask)".r,
          "(input_type)(.*)(ids)".r,
          "(bert)(.*)(encoder)".r)

      case "HF" =>
        Array(
          "(input)(.*)(ids)".r,
          "(attention)(.*)(mask)".r,
          "(token_type)(.*)(ids)".r,
          "(StatefulPartitionedCall)".r)

      case _ => throw new Exception("Unknown model provider! Please provide one in between JSL, TF2 or HF.")
    }
    referenceKeys
  }

  /** Convert signatures key names to adopted naming conventions for BERT keys mapping */
  def toAdoptedKeys(keyName: String) = {
    keyName match {
      case "input_ids" | "input_word_ids" => TokenIds.key
      case "input_mask" | "attention_mask" => MaskIds.key
      case "segment_ids" | "input_type_ids" | "token_type_ids" => SegmentIds.key
      case "sequence_output" | "bert_encoder" | "last_hidden_state" => Embeddings.key
      case "pooled_output" => SentenceEmbeddings.key
      case k => k
    }
  }
}
