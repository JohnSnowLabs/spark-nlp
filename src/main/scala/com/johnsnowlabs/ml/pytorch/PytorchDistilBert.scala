package com.johnsnowlabs.ml.pytorch

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}

class PytorchDistilBert(val pytorchWrapper: PytorchWrapper,
                        val sentenceStartTokenId: Int,
                        val sentenceEndTokenId: Int,
                        vocabulary: Map[String, Int])
  extends Serializable with PytorchTransformer {

  override protected val sentencePadTokenId: Int = 0

  override def tokenizeWithAlignment(tokenizedSentences: Seq[TokenizedSentence], caseSensitive: Boolean,
                                     maxSentenceLength: Int): Seq[WordpieceTokenizedSentence] = {

    val basicTokenizer = new BasicTokenizer(caseSensitive)
    val encoder = new WordpieceEncoder(vocabulary)

    tokenizedSentences.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens = tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map { token =>
        val content = if (caseSensitive) token.token else token.token.toLowerCase()
        val sentenceBegin = token.begin
        val sentenceEnd = token.end
        val sentenceIndex = tokenIndex.sentenceIndex
        val result = basicTokenizer.tokenize(Sentence(content, sentenceBegin, sentenceEnd, sentenceIndex))
        if (result.nonEmpty) result.head else IndexedToken("")
      }
      val wordpieceTokens = bertTokens.flatMap(token => encoder.encode(token)).take(maxSentenceLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  override def findIndexedToken(tokenizedSentences: Seq[TokenizedSentence], tokenWithEmbeddings: TokenPieceEmbeddings,
                                sentence: (WordpieceTokenizedSentence, Int)): Option[IndexedToken] = {

    val originalTokensWithEmbeddings = tokenizedSentences(sentence._2).indexedTokens.find(
      p => p.begin == tokenWithEmbeddings.begin)

    originalTokensWithEmbeddings

  }

}
