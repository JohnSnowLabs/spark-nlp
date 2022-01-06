package com.johnsnowlabs.ml.pytorch

import com.johnsnowlabs.ml.tensorflow.sentencepiece.{SentencePieceWrapper, SentencepieceEncoder}
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, TokenPieceEmbeddings, TokenizedSentence, WordpieceTokenizedSentence}

class PytorchAlbert(val pytorchWrapper: PytorchWrapper,
                    val sentencePieceWrapper: SentencePieceWrapper)
  extends Serializable with PytorchTransformer {

  // keys representing the input and output tensors of the ALBERT model
  override protected val sentenceStartTokenId: Int = sentencePieceWrapper.getSppModel.pieceToId("[CLS]")
  override protected val sentenceEndTokenId: Int = sentencePieceWrapper.getSppModel.pieceToId("[SEP]")
  override protected val sentencePadTokenId: Int = sentencePieceWrapper.getSppModel.pieceToId("[pad]")

  private val sentencePieceDelimiterId = sentencePieceWrapper.getSppModel.pieceToId("▁")

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

  override def findIndexedToken(tokenizedSentences: Seq[TokenizedSentence], tokenWithEmbeddings: TokenPieceEmbeddings,
                                sentence: (WordpieceTokenizedSentence, Int)): Option[IndexedToken] = {

    val originalTokensWithEmbeddings = tokenizedSentences(sentence._2).indexedTokens.find(
      p => p.begin == tokenWithEmbeddings.begin && tokenWithEmbeddings.isWordStart)

    originalTokensWithEmbeddings
  }

}
