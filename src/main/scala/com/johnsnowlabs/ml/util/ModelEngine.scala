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

package com.johnsnowlabs.ml.util

sealed trait ModelEngine

final case object TensorFlow extends ModelEngine {
  val name = "tensorflow"
  val modelName = "saved_model.pb"
}
final case object PyTorch extends ModelEngine {
  val name = "pytorch"
}

final case object ONNX extends ModelEngine {
  val name = "onnx"
  val modelName = "model.onnx"
  val encoderModel = "encoder_model.onnx"
  val decoderModel = "decoder_model.onnx"
  val decoderWithPastModel = "decoder_with_past_model.onnx"
}
final case object Openvino extends ModelEngine {
  val name = "openvino"
  val modelXml = "saved_model.xml"
  val modelBin = "saved_model.bin"
}

final case object Unknown extends ModelEngine {
  val name = "unk"
}
