package com.johnsnowlabs.ml.onnx

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}

import scala.collection.mutable.ArrayBuffer

/** Class to manage the creation of ONNX Tensors (WIP).
  *
  * Tensors that do not belong to a [[OrtSession]], will need to be explicitly closed. This class
  * manages created tensors and can free all of them at once.
  *
  * The [[OrtEnvironment]] exists at most once, so all TensorResources will share the same one.
  */
class TensorResources extends AutoCloseable {
  private val tensors = ArrayBuffer[OnnxTensor]()
  private val env = OrtEnvironment.getEnvironment()
  def createTensor(data: Any): OnnxTensor = {
    val tensor = OnnxTensor.createTensor(env, data)
    tensors.append(tensor)
    tensor
  }

  def clearTensors(): Unit = {
    tensors.foreach(_.close())
    tensors.clear()
  }

  override def close(): Unit = clearTensors()
}

object TensorResources {

  def getOnnxTensor(result: OrtSession.Result, key: String): OnnxTensor =
    result.get(key).get().asInstanceOf[OnnxTensor]
  object implicits {
    implicit class OnnxSessionResult(result: OrtSession.Result) {
      def getOnnxTensor(key: String): OnnxTensor =
        TensorResources.getOnnxTensor(result, key)

      def getOnnxTensors(keys: Array[String]): Map[String, OnnxTensor] = {
        keys.map { key =>
          (key, TensorResources.getOnnxTensor(result, key))
        }.toMap
      }

      def getFloatArray(key: String): Array[Float] = Option(
        TensorResources.getOnnxTensor(result, key).getFloatBuffer) match {
        case Some(floats) =>
          val floatArray = floats.array()
          floatArray
        case None => throw new IllegalStateException("Could not extract floats from OnnxTensor.")
      }

    }

  }

}
