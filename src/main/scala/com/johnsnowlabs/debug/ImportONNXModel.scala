package com.johnsnowlabs.debug

import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import com.johnsnowlabs.nlp.annotators.common.TokenizedSentence

import scala.collection.JavaConverters._

class ImportONNXModel(modelPath: String, tokenizer: TransformersTokenizer) {

  private val (ortSession, ortEnv) = {
    val env = OrtEnvironment.getEnvironment()
    val sessionOptions = new SessionOptions()
    val session = env.createSession(modelPath, sessionOptions)
    (session, env)
  }

  def computeLogitsWithContextRegular(
     batch: Seq[Array[Int]],
     maxSentenceLength: Int): (Array[Float], Array[Float]) = {
    // [nb of encoded sentences , maxSentenceLength]
    val tokenTensors =
      OnnxTensor.createTensor(ortEnv, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        ortEnv,
        batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

    val segmentTensors =
      OnnxTensor.createTensor(ortEnv, batch.map(x => Array.fill(maxSentenceLength)(0L)).toArray)

    val inputs =
      Map(
        "input_ids" -> tokenTensors,
        "attention_mask" -> maskTensors,
        "token_type_ids" -> segmentTensors).asJava

    try {
      val output = ortSession.run(inputs)
      println("debugging outputs")
      try {
        val startLogits = output
          .get("start_logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        val endLogits = output
          .get("end_logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        tokenTensors.close()
        maskTensors.close()
        segmentTensors.close()

        (startLogits, endLogits)
      } finally if (output != null) output.close()
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        println("Exception: ", e)
        // Rethrow the exception to propagate it further
        throw e
    }
  }

  def computeLogitsWithContext(
                                batch: Seq[Array[Int]],
                                maxSentenceLength: Int): Array[Float] = {
    // Add batch dimension by wrapping the arrays in another array (batch size of 1)
    val tokenTensors =
      OnnxTensor.createTensor(ortEnv, Array(batch.map(x => x.map(_.toLong)).toArray))
    val maskTensors =
      OnnxTensor.createTensor(
        ortEnv,
        Array(batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)
      )
    val segmentTensors =
      OnnxTensor.createTensor(ortEnv, Array(batch.map(_ => Array.fill(maxSentenceLength)(0L)).toArray))

    val inputs =
      Map(
        "input_ids" -> tokenTensors,
        "attention_mask" -> maskTensors,
        "token_type_ids" -> segmentTensors).asJava

    try {
      val info = ortSession.getInputInfo
      val metadata = ortSession.getMetadata
      val inputInfo = ortSession.getInputInfo
      val outputInfo = ortSession.getOutputInfo
      val outputNames = ortSession.getOutputNames
      val inputNames = ortSession.getInputNames
      val output = ortSession.run(inputs)
      println("debugging outputs")
      try {

        val logits = output
          .get("output")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        tokenTensors.close()
        maskTensors.close()
        segmentTensors.close()

        val scores = calculateSoftmax(logits)

        logits
      } finally if (output != null) output.close()
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        println("Exception: ", e)
        // Rethrow the exception to propagate it further
        throw e
    }
  }


  def computeLogits(
      tokenizedSentences: Seq[TokenizedSentence],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean) = {

    val wordPieceTokenizedSentences =
      tokenizer.tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)
    /*Run calculation by batches*/
    wordPieceTokenizedSentences.zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = tokenizer.encode(batch, maxSentenceLength)
        val logits = predict(encoded)
        logits
      }
      .toSeq
  }



  def predict(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    val batchLength = batch.length
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max

    val rawScores = ortSessionRun(batch, maxSentenceLength)

    val dim = rawScores.length / (batchLength * maxSentenceLength)
    val batchScores: Array[Array[Array[Float]]] = rawScores
      .grouped(dim)
      .map(scores => calculateSoftmax(scores))
      .toArray
      .grouped(maxSentenceLength)
      .toArray

    batchScores
  }

  private def calculateSoftmax(scores: Array[Float]): Array[Float] = {
    val exp = scores.map(x => math.exp(x))
    exp.map(x => x / exp.sum).map(_.toFloat)
  }

  def ortSessionRun(batch: Seq[Array[Int]], maxSentenceLength: Int) = {
    val tokenTensors =
      OnnxTensor.createTensor(ortEnv, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        ortEnv,
        batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

    val segmentTensors =
      OnnxTensor.createTensor(ortEnv, batch.map(x => Array.fill(maxSentenceLength)(0L)).toArray)

    val inputs =
      Map(
        "input_ids" -> tokenTensors,
        "attention_mask" -> maskTensors,
        "token_type_ids" -> segmentTensors).asJava

    try {
      val results = ortSession.run(inputs)
      try {
        val logits = results
          .get("logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        logits
      } finally if (results != null) results.close()
    } catch {
      case e: Exception =>
        // Handle exceptions by logging or other means.
        e.printStackTrace()
        Array.empty[Float] // Return an empty array or appropriate error handling
    } finally {
      // Close tensors outside the try-catch to avoid repeated null checks.
      // These resources are initialized before the try-catch, so they should be closed here.
      tokenTensors.close()
      maskTensors.close()
      segmentTensors.close()
    }
  }

}
