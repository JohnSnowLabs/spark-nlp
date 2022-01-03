package com.johnsnowlabs.ml.pytorch

import ai.djl.{Device, Model}
import ai.djl.ndarray.NDList
import ai.djl.pytorch.engine.PtModel
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{SentencePieceWrapper, SentencepieceEncoder}
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, TokenPieceEmbeddings, TokenizedSentence, WordpieceTokenizedSentence}
import com.johnsnowlabs.nlp.embeddings.TransformerEmbeddings

import java.io.ByteArrayInputStream

class PytorchAlbert(val pytorchWrapper: PytorchWrapper, val sentencePieceWrapper: SentencePieceWrapper)
  extends Serializable
  with Translator[Array[Array[Int]], Array[Array[Float]]]
  with TransformerEmbeddings {

  // keys representing the input and output tensors of the ALBERT model
  override protected val sentenceStartTokenId: Int = sentencePieceWrapper.getSppModel.pieceToId("[CLS]")
  override protected val sentenceEndTokenId: Int = sentencePieceWrapper.getSppModel.pieceToId("[SEP]")
  override protected val sentencePadTokenId: Int = sentencePieceWrapper.getSppModel.pieceToId("[pad]")

  private val sentencePieceDelimiterId = sentencePieceWrapper.getSppModel.pieceToId("▁")

  private lazy val predictor = {
    val modelInputStream = new ByteArrayInputStream(pytorchWrapper.modelBytes)
    val device = Device.cpu() //TODO: Check with gpu
    val model = Model.newInstance("albert-model", device)

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

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val output = predictor.predict(batch.toArray)
    val dimension = output.head.head.toInt
    val allEncoderLayers = output.last
    val predictedEmbeddings = allEncoderLayers
      .grouped(dimension).toArray
      .grouped(maxSentenceLength).toArray

    val emptyVector = Array.fill(dimension)(0f)
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

  override def findIndexedToken(tokenizedSentences: Seq[TokenizedSentence], tokenWithEmbeddings: TokenPieceEmbeddings,
                                sentence: (WordpieceTokenizedSentence, Int)): Option[IndexedToken] = {

    val originalTokensWithEmbeddings = tokenizedSentences(sentence._2).indexedTokens.find(
      p => p.begin == tokenWithEmbeddings.begin && tokenWithEmbeddings.isWordStart)

    originalTokensWithEmbeddings
  }


  override def getBatchifier: Batchifier = {
    Batchifier.fromString("none")
  }

  override def processInput(ctx: TranslatorContext, input: Array[Array[Int]]): NDList = {
    val manager = ctx.getNDManager
    val array = manager.create(input)
    new NDList(array)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): Array[Array[Float]] = {
    val dimension = Array(list.get(0).getShape.get(2).toFloat)
    val allEncoderLayers = list.get(0).toFloatArray

    Array(dimension, allEncoderLayers)
  }

}
