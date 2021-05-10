package com.johnsnowlabs.ml.tensorflow.sign

import scala.util.matching.Regex


/**
 * Based on BERT SavedModel reference, for instance:
 *
 * signature_def['serving_default']:
 * The given SavedModel SignatureDef contains the following input(s):
 *
 * inputs['attention_mask'] tensor_info:
 * dtype: DT_INT32
 * shape: (-1, -1)
 * name: serving_default_attention_mask:0
 *
 * inputs['input_ids'] tensor_info:
 * dtype: DT_INT32
 * shape: (-1, -1)
 * name: serving_default_input_ids:0
 *
 * inputs['token_type_ids'] tensor_info:
 * dtype: DT_INT32
 * shape: (-1, -1)
 * name: serving_default_token_type_ids:0
 *
 * The given SavedModel SignatureDef contains the following output(s):
 *
 * outputs['last_hidden_state'] tensor_info:
 * dtype: DT_FLOAT
 * shape: (-1, -1, 768)
 * name: StatefulPartitionedCall:0
 *
 * outputs['pooler_output'] tensor_info:
 * dtype: DT_FLOAT
 * shape: (-1, 768)
 * name: StatefulPartitionedCall:1
 *
 * Method name is: tensorflow/serving/predict*
 */


object ModelSignatureConstants {

  sealed trait TFInfoDescriptor {
    protected val key: String
  }

  case object Name extends TFInfoDescriptor {
    override val key: String = "name"
  }

  case object DType extends TFInfoDescriptor {
    override val key: String = "dtype"
  }

  case object DimCount extends TFInfoDescriptor {
    override val key: String = "dim_count"
  }

  case object ShapeDimList extends TFInfoDescriptor {
    override val key: String = "shape_dim_list"
  }

  case object SerializedSize extends TFInfoDescriptor {
    override val key: String = "serialized_size"
  }

  sealed trait TFInfoNameMapper {
    protected val key: String
    protected val value: String
  }

  case object InputIds extends TFInfoNameMapper {
    override val key: String = "input_ids"
    override val value: String = "input_ids:0"
  }

  case object InputIdsV1 extends TFInfoNameMapper {
    override val key: String = "input_ids"
    override val value: String = "input_ids:0"
  }

  case object AttentionMask extends TFInfoNameMapper {
    override val key: String = "attention_mask"
    override val value: String = "attention_mask:0"
  }

  case object AttentionMaskV1 extends TFInfoNameMapper {
    override val key: String = "attention_mask"
    override val value: String = "input_mask:0"
  }

  case object TokenTypeIds extends TFInfoNameMapper {
    override val key: String = "token_type_ids"
    override val value: String = "token_type_ids:0"
  }

  case object TokenTypeIdsV1 extends TFInfoNameMapper {
    override val key: String = "token_type_ids"
    override val value: String = "segment_ids:0"
  }

  case object LastHiddenState extends TFInfoNameMapper {
    override val key: String = "last_hidden_state"
    override val value: String = "StatefulPartitionedCall:0"
  }

  case object LastHiddenStateV1 extends TFInfoNameMapper {
    override val key: String = "last_hidden_state"
    override val value: String = "sequence_output:0"
  }

  case object PoolerOutput extends TFInfoNameMapper {
    override val key: String = "pooler_output"
    override val value: String = "StatefulPartitionedCall:1"
  }

  case object PoolerOutputV1 extends TFInfoNameMapper {
    override val key: String = "pooler_output"
    override val value: String = "pooled_output:0"
  }

  /** Retrieve signature patterns for a given provider
   *
   * @param modelProvider : the provider library that built the model and the signatures
   * @return reference keys array of signature patterns for a given provider
   * */
  def getSignaturePatterns(modelProvider: String): Array[Regex] = {
    val referenceKeys = modelProvider match {
      case "TF1" =>
        Array(
          "(input)(.*)(ids)".r,
          "(input)(.*)(mask)".r,
          "(segment)(.*)(ids)".r,
          "(sequence)(.*)(output)".r,
          "(pooled)(.*)(output)".r)

      case "TF2" =>
        Array(
          // first possible keys
          "(input_word)(.*)(ids)".r,
          "(input)(.*)(mask)".r,
          "(input_type)(.*)(ids)".r,
          "(bert)(.*)(encoder)".r,
          // second possible keys
          "(input)(.*)(ids)".r,
          "(attention)(.*)(mask)".r,
          "(token_type)(.*)(ids)".r,
          "(last_hidden_state)".r)

      case _ => throw new Exception("Unknown model provider! Please provide one in between TF1 or TF2.")
    }
    referenceKeys
  }

  /** Convert signatures key names to normalize naming conventions */
  def toAdoptedKeys(keyName: String): String = {
    keyName match {
      case "input_ids" | "input_word_ids" => InputIds.key
      case "input_mask" | "attention_mask" => AttentionMask.key
      case "segment_ids" | "input_type_ids" | "token_type_ids" => TokenTypeIds.key
      case "sequence_output" | "bert_encoder" | "last_hidden_state" => LastHiddenState.key
      case "pooler_output" => PoolerOutput.key
      case k => k
    }
  }
}
