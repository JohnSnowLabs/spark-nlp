package com.johnsnowlabs.ml.pytorch

import ai.djl.ndarray.NDList
import ai.djl.pytorch.engine.PtModel
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}
import ai.djl.{Device, Model}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.embeddings.TransformerEmbeddings

import java.io.ByteArrayInputStream

class PytorchBert(val pytorchWrapper: PytorchWrapper,
                  val sentenceStartTokenId: Int,
                  val sentenceEndTokenId: Int,
                  vocabulary: Map[String, Int]) extends Serializable
    with Translator[Array[Array[Int]], Array[Array[Float]]]
    with TransformerEmbeddings {

  override protected val sentencePadTokenId: Int = 0

  private lazy val predictor = {
    val modelInputStream = new ByteArrayInputStream(pytorchWrapper.modelBytes)
    val device = Device.cpu() //TODO: Check with gpu
    val model = Model.newInstance("bert-model", device)

    val pyTorchModel: PtModel = model.asInstanceOf[PtModel]
    pyTorchModel.load(modelInputStream)

    pyTorchModel.newPredictor(this)
  }

  override def tokenizeWithAlignment(tokenizedSentences: Seq[TokenizedSentence], caseSensitive: Boolean,
                                     maxSentenceLength: Int): Seq[WordpieceTokenizedSentence] = {

    //TODO: Check how many Transformers implement this in the same way.
    // Implement the same: Bert, Distilbert
    // Does NOT implement the same: Albert
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

  override def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    //TODO: Check how many Transformers implement this in the same way.
    // Implement the same: Bert, Albert, Distilbert
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
