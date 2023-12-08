/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.ml.tensorflow.sign

import scala.util.matching.Regex

/** Based on BERT SavedModel reference, for instance:
  *
  * signature_def['serving_default']: The given SavedModel SignatureDef contains the following
  * input(s):
  *
  * inputs['attention_mask'] tensor_info: dtype: DT_INT32 shape: (-1, -1) name:
  * serving_default_attention_mask:0
  *
  * inputs['input_ids'] tensor_info: dtype: DT_INT32 shape: (-1, -1) name:
  * serving_default_input_ids:0
  *
  * inputs['token_type_ids'] tensor_info: dtype: DT_INT32 shape: (-1, -1) name:
  * serving_default_token_type_ids:0
  *
  * The given SavedModel SignatureDef contains the following output(s):
  *
  * outputs['last_hidden_state'] tensor_info: dtype: DT_FLOAT shape: (-1, -1, 768) name:
  * StatefulPartitionedCall:0
  *
  * outputs['pooler_output'] tensor_info: dtype: DT_FLOAT shape: (-1, 768) name:
  * StatefulPartitionedCall:1
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

  case object EncoderInputIds extends TFInfoNameMapper {
    override val key: String = "encoder_input_ids"
    override val value: String = "encoder_encoder_input_ids:0"
  }

  case object EncoderAttentionMask extends TFInfoNameMapper {
    override val key: String = "encoder_attention_mask"
    override val value: String = "encoder_encoder_attention_mask:0"
  }

  case object EncoderOutput extends TFInfoNameMapper {
    override val key: String = "last_hidden_state"
    override val value: String = "StatefulPartitionedCall_1:0"
  }

  case object DecoderInputIds extends TFInfoNameMapper {
    override val key: String = "decoder_input_ids"
    override val value: String = "decoder_decoder_input_ids:0"
  }

  case object DecoderEncoderInputIds extends TFInfoNameMapper {
    override val key: String = "encoder_state"
    override val value: String = "decoder_encoder_state:0"
  }

  case object DecoderEncoderAttentionMask extends TFInfoNameMapper {
    override val key: String = "decoder_encoder_attention_mask"
    override val value: String = "decoder_decoder_encoder_attention_mask:0:0"
  }

  case object DecoderAttentionMask extends TFInfoNameMapper {
    override val key: String = "decoder_attention_mask"
    override val value: String = "decoder_decoder_attention_mask:0"
  }

  case object DecoderOutput extends TFInfoNameMapper {
    override val key: String = "output_0"
    override val value: String = "StatefulPartitionedCall:0"
  }

  case object DecoderCachedOutput extends TFInfoNameMapper {
    override val key: String = "output_0"
    override val value: String = "StatefulPartitionedCall_1:2"
  }

  case object DecoderInitOutputCache1Key extends TFInfoNameMapper {
    override val key: String = "output_cache1"
    override val value: String = "StatefulPartitionedCall_1:0"
  }

  case object DecoderInitOutputCache2Key extends TFInfoNameMapper {
    override val key: String = "output_cache2"
    override val value: String = "StatefulPartitionedCall_1:1"
  }

  case object DecoderCachedInputIdsKey extends TFInfoNameMapper {
    override val key: String = "decoder_cached_input_ids"
    override val value: String = "decoder_cached_decoder_input_ids:0"
  }

  case object DecoderCachedEncoderStateKey extends TFInfoNameMapper {
    override val key: String = "decoder_cached_encoder_state"
    override val value: String = "decoder_cached_encoder_state:0"
  }

  case object DecoderCachedEncoderAttentionKey extends TFInfoNameMapper {
    override val key: String = "decoder_cached_encoder_attention"
    override val value: String = "decoder_cached_encoder_attention_mask:0"
  }

  case object DecoderCachedCache1Key extends TFInfoNameMapper {
    override val key: String = "decoder_cached_cache1"
    override val value: String = "decoder_cached_cache1:0"
  }

  case object DecoderCachedCache2Key extends TFInfoNameMapper {
    override val key: String = "decoder_cached_cache2"
    override val value: String = "decoder_cached_cache2:0"
  }

  case object DecoderCachedOutputKey extends TFInfoNameMapper {
    override val key: String = "decoder_cached_output"
    override val value: String = "StatefulPartitionedCall:2"
  }

  case object DecoderCachedOutputCache1Key extends TFInfoNameMapper {
    override val key: String = "decoder_cached_output_cache1"
    override val value: String = "StatefulPartitionedCall:0"
  }

  case object DecoderCachedOutputCache2Key extends TFInfoNameMapper {
    override val key: String = "decoder_cached_output_cache2"
    override val value: String = "StatefulPartitionedCall:1"
  }

  case object LogitsOutput extends TFInfoNameMapper {
    override val key: String = "logits"
    override val value: String = "StatefulPartitionedCall:0"
  }

  case object EndLogitsOutput extends TFInfoNameMapper {
    override val key: String = "end_logits"
    override val value: String = "StatefulPartitionedCall:0"
  }

  case object StartLogitsOutput extends TFInfoNameMapper {
    override val key: String = "start_logits"
    override val value: String = "StatefulPartitionedCall:1"
  }

  case object TapasLogitsOutput extends TFInfoNameMapper {
    override val key: String = "logits"
    override val value: String = "StatefulPartitionedCall:0"
  }

  case object TapasLogitsAggregationOutput extends TFInfoNameMapper {
    override val key: String = "logits_aggregation"
    override val value: String = "StatefulPartitionedCall:1"
  }

  case object PixelValuesInput extends TFInfoNameMapper {
    override val key: String = "pixel_values"
    override val value: String = "pixel_values:0"
  }

  case object AudioValuesInput extends TFInfoNameMapper {
    override val key: String = "input_values"
    override val value: String = "input_values:0"
  }

  case object CachedEncoderOutput extends TFInfoNameMapper {
    override val key: String = "last_hidden_state"
    override val value: String = "StatefulPartitionedCall_2:0"
  }

  case object CachedDecoderInputIds extends TFInfoNameMapper {
    override val key: String = "decoder_input_ids"
    override val value: String = "decoder_cached_decoder_input_ids:0"
  }

  case object CachedDecoderEncoderInputIds extends TFInfoNameMapper {
    override val key: String = "encoder_state"
    override val value: String = "decoder_cached_encoder_state:0"
  }

  case object CachedDecoderEncoderAttentionMask extends TFInfoNameMapper {
    override val key: String = "decoder_encoder_attention_mask"
    override val value: String = "decoder_cached_decoder_encoder_attention_mask:0"
  }

  case object CachedDecoderInputCache1 extends TFInfoNameMapper {
    override val key: String = "cache1"
    override val value: String = "decoder_cached_cache1:0"
  }

  case object CachedDecoderInputCache2 extends TFInfoNameMapper {
    override val key: String = "cache2"
    override val value: String = "decoder_cached_cache2:0"
  }

  case object CachedOutput1 extends TFInfoNameMapper {
    override val key: String = "cache1_out"
    override val value: String = "StatefulPartitionedCall:0"
  }

  case object CachedOutPut2 extends TFInfoNameMapper {
    override val key: String = "cache2_out"
    override val value: String = "StatefulPartitionedCall:1"
  }

  case object CachedLogitsOutput extends TFInfoNameMapper {
    override val key: String = "logits"
    override val value: String = "StatefulPartitionedCall:2"
  }

  case object InitDecoderInputIds extends TFInfoNameMapper {
    override val key: String = "decoder_input_ids_init"
    override val value: String = "decoder_init_decoder_input_ids_init:0"
  }

  case object InitDecoderEncoderInputIds extends TFInfoNameMapper {
    override val key: String = "encoder_state_init"
    override val value: String = "decoder_init_encoder_state_init:0"
  }

  case object InitDecoderEncoderAttentionMask extends TFInfoNameMapper {
    override val key: String = "decoder_encoder_attention_mask_init"
    override val value: String = "decoder_init_decoder_encoder_attention_mask_init:0"
  }

  case object InitCachedOutput1 extends TFInfoNameMapper {
    override val key: String = "cache1_out_init"
    override val value: String = "StatefulPartitionedCall_1:0"
  }

  case object InitCachedOutPut2 extends TFInfoNameMapper {
    override val key: String = "cache2_out_init"
    override val value: String = "StatefulPartitionedCall_1:1"
  }

  case object InitLogitsOutput extends TFInfoNameMapper {
    override val key: String = "logits_init"
    override val value: String = "StatefulPartitionedCall_1:2"
  }

  case object EncoderContextMask extends TFInfoNameMapper {
    override val key: String = "encoder_context_mask"
    override val value: String = "encoder_encoder_context_mask:0"
  }

  /** Retrieve signature patterns for a given provider
    *
    * @param modelProvider
    *   : the provider library that built the model and the signatures
    * @return
    *   reference keys array of signature patterns for a given provider
    */
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

      case _ =>
        throw new Exception("Unknown model provider! Please provide one in between TF1 or TF2.")
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
