package com.johnsnowlabs.ml.tensorflow.wrap

import org.tensorflow.ndarray.buffer.ByteDataBuffer


case class ModelSignature(operation: String, value: String, matchingPatterns: List[String])

case class Variables(variables: Array[Array[Byte]], index: Array[Byte])

case class VariablesTfIo(variables: ByteDataBuffer, index: ByteDataBuffer)