package com.johnsnowlabs.ml.ai.seq2seq

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import com.johnsnowlabs.ml.ai.util.Generation.Generate
import com.johnsnowlabs.ml.onnx.TensorResources.implicits.OnnxSessionResult
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.util.LoadExternalModel.loadTextAsset
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BlenderBotTokenizer, BpeTokenizer}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.intel.openvino.InferRequest
import org.tensorflow.{Session, Tensor}

import java.io.File
import scala.collection.JavaConverters._

class BlenderBot extends Generate {

  val mainModelPath = "/media/danilo/Data/Danilo/JSL/models/transformers"
  val onnxModelPath = s"$mainModelPath/onnx/blenderbot/blenderbot.onnx"
  val tokenizerModelPath = s"$mainModelPath/tf/blenderbot-400M-distill-tokenizer"
  val vocabSize = 8008
  val maxOutputLength = 25 // Set to the maximum sequence length expected by the model
  val paddingTokenId = 0
  val bosTokenId = 1

  def testOutput(): Unit = {
    val bpeTokenizer = getBlenderBotTokenizer
//    val array: Array[Int] = Array(
//      6950, 505 //hello there
//    )

    val array: Array[Int] = Array(
      47, 921, 86
    )

    val result = decode(Array(array), bpeTokenizer)
    println(s"replyText: ${result.head}")
  }

  def tag(text: String): Unit = {

    val bpeTokenizer = getBlenderBotTokenizer
    val inputIds = encode(text, bpeTokenizer)
    val attentionMask: Array[Array[Long]] = inputIds.map(inputId => inputId.map(x => if (x == 0L) 0L else 1L)).toArray
    val decoderInputIds: Array[Array[Int]] =  Array(Array.fill(inputIds.head.length)(bosTokenId)) //TODO: bosTokenId??

    val (onnxSession, onnxEnvironment) = getOnnxSession

    // Print input and output details
//    val inputsInfo = onnxSession.getInputInfo
//    val outputsInfo = onnxSession.getOutputInfo
//
//    println("Model Inputs:")
//    inputsInfo.forEach { case (name, info) =>
//      println(s"Name: $name, Info: ${info.getInfo}")
//    }
//
//    println("Model Outputs:")
//    outputsInfo.forEach { case (name, info) =>
//      println(s"Name: $name, Info: ${info.getInfo}")
//    }

    val attentionMaskTensor = OnnxTensor.createTensor(onnxEnvironment, attentionMask)
    val decoderEncoderStateTensors = OnnxTensor.createTensor(onnxEnvironment, Array(0))


    val outputSeq = generate(
      inputIds = inputIds,
      decoderEncoderStateTensors = Right(decoderEncoderStateTensors),
      encoderAttentionMaskTensors = Right(attentionMaskTensor),
      decoderInputs = decoderInputIds,
      maxOutputLength = maxOutputLength,
      minOutputLength = 5,
      doSample = true,
      beamSize = 2,
      numReturnSequences = 1,
      temperature = 0.7,
      topK = 50,
      topP = 0.95,
      repetitionPenalty = 2.5,
      noRepeatNgramSize = 2,
      vocabSize = vocabSize,
      eosTokenId = 2,
      paddingTokenId = paddingTokenId,
      randomSeed = None,
      ignoreTokenIds = Array(),
      session = Right((onnxEnvironment, onnxSession)),
      applySoftmax = false,
      ovInferRequest = None,
      stopTokenIds = Array() //
    )
    println(s"outputSeq: ${outputSeq.head.length}")
    println(s"outputSeq: ${outputSeq.head.mkString(" ")}")
    val replyText = decode(outputSeq, bpeTokenizer)
    println(s"replyText: ${replyText.head}")
  }

  def getBlenderBotTokenizer: BlenderBotTokenizer = {
    val vocabs = loadTextAsset(tokenizerModelPath, "vocab.txt").zipWithIndex.toMap
    val bytePairs = loadTextAsset(tokenizerModelPath, "merges.txt")
      .map(_.split(" "))
      .filter(w => w.length == 2)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex
      .toMap

    BpeTokenizer
      .forModel("blenderbot", merges = bytePairs, vocab = vocabs)
      .asInstanceOf[BlenderBotTokenizer]
  }

  def encode(text: String, bpeTokenizer: BlenderBotTokenizer): Seq[Array[Int]] = {
    val sentence = Sentence(content = text, start = 0, end = text.length, index = 0, None)

    Seq(bpeTokenizer.tokenize(sentence)
      .map(bpeTokenizer.encode).flatMap(_.map(_.pieceId)))
  }

  def decode(sentences: Array[Array[Int]], bpeTokenizer: BlenderBotTokenizer): Seq[String] = {
    sentences.map(sentence => bpeTokenizer.decodeTokens(sentence))
  }

  private def getOnnxSession: (OrtSession, OrtEnvironment) = {
    val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions
    ResourceHelper.spark.sparkContext.addFile(onnxModelPath)
    val onnxBlenderBotFileName = Some(new File(onnxModelPath).getName)
    val blenderBotOnnxWrapper = new OnnxWrapper(onnxBlenderBotFileName, None)

    blenderBotOnnxWrapper.getSession(onnxSessionOptions)
  }

  /** Calls the model and returns the output logits.
   *
   * @param encoderInputIds
   * Input IDs for the Encoder
   * @param decoderInputIds
   * Input IDs for the Decoder
   * @param decoderEncoderStateTensors
   * Tensor of encoded input for the decoder
   * @param encoderAttentionMaskTensors
   * Tensor for encoder attention mask
   * @param maxLength
   * Max length of the input
   * @param session
   * Tensorflow Session
   * @return
   * Logits for the input
   */
  override def getModelOutput(
     encoderInputIds: Seq[Array[Int]],
     decoderInputIds: Seq[Array[Int]],
     decoderEncoderStateTensors: Either[Tensor, OnnxTensor],
     encoderAttentionMaskTensors: Either[Tensor, OnnxTensor],
     maxLength: Int, session: Either[Session, (OrtEnvironment, OrtSession)],
     ovInferRequest: Option[InferRequest]): Array[Array[Float]] = {

    // Expand encoderInputIds and attentionMask based on the size of decoderInputIds
    val sequenceSize = decoderInputIds.head.length
    val beamSize = decoderInputIds.length  // Beam size inferred from decoder input size
    println(s"sequenceSize: $sequenceSize")
    println(s"beamSize: $beamSize")

    // Repeat encoderInputIds and attentionMask to match the beam size
    val encoderInputIdsLong = encoderInputIds.map(inputId => inputId.map(_.toLong)).toArray

    // Expand encoderInputIds to match the size of decoderInputIds
    // If decoderInputIds grows, we need to "repeat" encoderInputIds to match it
    val encoderInputIdsExpanded = encoderInputIdsLong.map(_.padTo(sequenceSize, paddingTokenId.toLong))  // Padding with 0 or whatever padding token is appropriate

    val (onnxEnvironment, onnxSession) = session.right.get

    val attentionMaskExpanded = encoderAttentionMaskTensors match {
      case Right(mask) =>
        val attentionMask = mask.getValue.asInstanceOf[Array[Array[Long]]] // Get the underlying attention mask
        val attentionMask1D = attentionMask.map(_.padTo(sequenceSize, paddingTokenId.toLong))
        Array.fill(beamSize)(attentionMask1D(0))
      case Left(_) => throw new IllegalArgumentException("Invalid attention mask tensor")
    }
    val attentionMaskTensorExpanded = OnnxTensor.createTensor(onnxEnvironment, attentionMaskExpanded)

    val decoderInputIdsLong = decoderInputIds.map(inputId => inputId.map(_.toLong)).toArray
    val inputIdsTensorExpanded = OnnxTensor.createTensor(onnxEnvironment, encoderInputIdsExpanded)

    val decoderInputIdsTensor = OnnxTensor.createTensor(onnxEnvironment, decoderInputIdsLong)

    val inputs =
      Map(
        "input_ids" -> inputIdsTensorExpanded,
        "attention_mask" -> attentionMaskTensorExpanded,
        "decoder_input_ids" -> decoderInputIdsTensor).asJava

    val sessionOutput = onnxSession.run(inputs)
    val logitsRaw = sessionOutput.getFloatArray("logits")

    val sequenceLength = encoderInputIdsLong.head.length
    val batchSizeEncoder = encoderInputIdsLong.length
    println(s"batchSizeEncoder: $batchSizeEncoder")

    val logits = (0 until batchSizeEncoder).map(i => {
      logitsRaw
        .slice(
          i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
          i * sequenceLength * vocabSize + sequenceLength * vocabSize)
    }).toArray

    println(s"logitsRaw size: ${logitsRaw.length}")
    println(s"logits size: ${logits.head.length}")
    logits
  }

}
