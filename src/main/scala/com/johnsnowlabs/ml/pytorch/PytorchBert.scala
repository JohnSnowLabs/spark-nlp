package com.johnsnowlabs.ml.pytorch

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.pytorch.engine.PtModel
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}
import com.johnsnowlabs.nlp.annotators.common._

import java.io.ByteArrayInputStream

class PytorchBert(val pytorchWrapper: PytorchWrapper, sentenceStartTokenId: Int, sentenceEndTokenId: Int)
  extends Translator[Array[Array[Int]], Array[Array[Array[Float]]]] with Serializable {

  private var maxSentenceLength: Option[Int] = None
  private var batchLength: Option[Int] = None
  private var dimension: Option[Int] = None

  private lazy val predictor = {
    val modelInputStream = new ByteArrayInputStream(pytorchWrapper.modelBytes)
    val pyTorchModel: PtModel = Model.newInstance("bert-model").asInstanceOf[PtModel]
    pyTorchModel.load(modelInputStream)

    pyTorchModel.newPredictor(this)
  }

  def calculateEmbeddings(sentences: Seq[WordpieceTokenizedSentence],
                          originalTokenSentences: Seq[TokenizedSentence],
                          batchSize: Int,
                          maxSentenceLength: Int,
                          caseSensitive: Boolean): Seq[WordpieceEmbeddingsSentence] = {
    /*Run embeddings calculation by batches*/
    sentences.zipWithIndex.grouped(batchSize).flatMap { batch =>
      val encoded = encode(batch, maxSentenceLength)
      val vectors = tag(encoded.toArray)

      /*Combine tokens and calculated embeddings*/
      batch.zip(vectors).map { case (sentence, tokenVectors) =>
        val tokenLength = sentence._1.tokens.length

        /*All wordpiece embeddings*/
        val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)

        /*Word-level and span-level alignment with Tokenizer
        https://github.com/google-research/bert#tokenization
        ### Input
        orig_tokens = ["John", "Johanson", "'s",  "house"]
        labels      = ["NNP",  "NNP",      "POS", "NN"]
        # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
        # orig_to_tok_map == [1, 2, 4, 6]*/

        val tokensWithEmbeddings = sentence._1.tokens.zip(tokenEmbeddings).flatMap {
          case (token, tokenEmbedding) =>
            val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
            val originalTokensWithEmbeddings = originalTokenSentences(sentence._2).indexedTokens.find(
              p => p.begin == tokenWithEmbeddings.begin && tokenWithEmbeddings.isWordStart).map {
              token =>
                val originalTokenWithEmbedding = TokenPieceEmbeddings(
                  TokenPiece(wordpiece = tokenWithEmbeddings.wordpiece,
                    token = if (caseSensitive) token.token else token.token.toLowerCase(),
                    pieceId = tokenWithEmbeddings.pieceId,
                    isWordStart = tokenWithEmbeddings.isWordStart,
                    begin = token.begin,
                    end = token.end
                  ),
                  tokenEmbedding
                )
                originalTokenWithEmbedding
            }
            originalTokensWithEmbeddings
        }

        WordpieceEmbeddingsSentence(tokensWithEmbeddings, sentence._2)
      }
    }.toSeq
  }

  /** Encode the input sequence to indexes IDs adding padding where necessary */
  def encode(sentences: Seq[(WordpieceTokenizedSentence, Int)], maxSequenceLength: Int): Seq[Array[Int]] = {
    val maxSentenceLength =
      Array(
        maxSequenceLength - 2,
        sentences.map { case (wpTokSentence, _) => wpTokSentence.tokens.length }.max).min

    sentences
      .map { case (wpTokSentence, _) =>
        val tokenPieceIds = wpTokSentence.tokens.map(t => t.pieceId)
        val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(0)

        Array(sentenceStartTokenId) ++ tokenPieceIds.take(maxSentenceLength) ++ Array(sentenceEndTokenId) ++ padding
      }
  }

  def tag(batch: Array[Array[Int]]): Array[Array[Array[Float]]] = {
    maxSentenceLength = Some(batch.map(encodedSentence => encodedSentence.length).max)
    batchLength = Some(batch.length)
    val predictedEmbeddings = predictor.predict(batch)
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
    val embeddings = allEncoderLayers.grouped(dimension.get).toArray.grouped(maxSentenceLength.get).toArray

    embeddings
  }

}
