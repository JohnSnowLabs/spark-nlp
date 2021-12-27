package com.johnsnowlabs.ml.pytorch

import ai.djl.{Device, Model}
import ai.djl.ndarray.NDList
import ai.djl.pytorch.engine.PtModel
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{SentencePieceWrapper, SentencepieceEncoder}
import com.johnsnowlabs.nlp.annotators.common.{TokenizedSentence, WordpieceTokenizedSentence}
import com.johnsnowlabs.nlp.embeddings.TransformerEmbeddings

import java.io.ByteArrayInputStream

class PytorchAlbert(val pytorchWrapper: PytorchWrapper, val sentencePieceWrapper: SentencePieceWrapper)
  extends Serializable
  with Translator[Array[Array[Int]], Array[Array[Array[Float]]]]
  with TransformerEmbeddings {

  // keys representing the input and output tensors of the ALBERT model
  override protected val sentenceStartTokenId: Int = sentencePieceWrapper.getSppModel.pieceToId("[CLS]")
  override protected val sentenceEndTokenId: Int = sentencePieceWrapper.getSppModel.pieceToId("[SEP]")
  override protected val sentencePadTokenId: Int = sentencePieceWrapper.getSppModel.pieceToId("[pad]")

  private val sentencePieceDelimiterId = sentencePieceWrapper.getSppModel.pieceToId("▁")

  private var maxSentenceLength: Option[Int] = None
  private var dimension: Option[Int] = None

  private lazy val predictor = {
    val modelInputStream = new ByteArrayInputStream(pytorchWrapper.modelBytes)
    val device = Device.cpu() //TODO: Check with gpu
    val model = Model.newInstance("albert-model", device)
    println(s"Device Id: ${model.getNDManager.getDevice.getDeviceId}")
    println(s"Device Type: ${model.getNDManager.getDevice.getDeviceType}")

    val pyTorchModel: PtModel = model.asInstanceOf[PtModel]
    pyTorchModel.load(modelInputStream)

    pyTorchModel.newPredictor(this)
  }

  override def tokenizeWithAlignment(tokenizedSentences: Seq[TokenizedSentence], caseSensitive: Boolean,
                                     maxSentenceLength: Int): Seq[WordpieceTokenizedSentence] = {
    val encoder = new SentencepieceEncoder(sentencePieceWrapper, caseSensitive, delimiterId = sentencePieceDelimiterId)

    val sentenceTokenPieces = tokenizedSentences.map { s =>
      val shrinkedSentence = s.indexedTokens.take(maxSentenceLength - 2)
      val wordpieceTokens = shrinkedSentence.flatMap(token => encoder.encode(token)).take(maxSentenceLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentenceTokenPieces
  }

  override def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    maxSentenceLength = Some(batch.map(encodedSentence => encodedSentence.length).max)
    val predictedEmbeddings = predictor.predict(batch.toArray)
    val emptyVector = Array.fill(dimension.get)(0f)

    batch.zip(predictedEmbeddings).map { case (ids, embeddings) =>
      if (ids.length > embeddings.length) {
        embeddings.take(embeddings.length - 1) ++
          Array.fill(embeddings.length - ids.length)(emptyVector) ++
          Array(embeddings.last)
      } else {
        embeddings
      }
    }
  }

  override def getBatchifier: Batchifier = {
    Batchifier.fromString("none")
  }

  override def processInput(ctx: TranslatorContext, input: Array[Array[Int]]): NDList = {
    val manager = ctx.getNDManager
    val array = manager.create(input)
    new NDList(array)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): Array[Array[Array[Float]]] = {
    dimension = Some(list.get(0).getShape.get(2).toInt)
    val allEncoderLayers = list.get(0).toFloatArray
    val embeddings = allEncoderLayers
      .grouped(dimension.get).toArray
      .grouped(maxSentenceLength.get).toArray

    embeddings
  }

}
