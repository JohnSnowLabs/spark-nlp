package com.johnsnowlabs.ml.tensorflow

import org.tensorflow.op.Scope
import org.tensorflow.op.core.Constant
import org.tensorflow.op.random.StatelessRandomNormal
import org.tensorflow.types.{TFloat32, TInt32}
import org.tensorflow._

import java.util

class TensorflowEmbeddingLookup(embeddingSize: Int, vocabularySize: Int) {

  val shape: Array[Int] = Array(vocabularySize, embeddingSize)

  def initializeTable(initializer: String = "normal", seed: Int = 1): Tensor[TFloat32] = {

    val eagerSession = EagerSession.create()
    val scope = new Scope(eagerSession)

    val embeddingsTable: Tensor[TFloat32] = initializer match {
      case "normal" => initializeRandomNormal(scope, seed)
      case _ => throw new IllegalArgumentException("Undefined initializer.")
    }

    embeddingsTable
  }

  def lookup(embeddingsTable: Tensor[TFloat32], indexes: Array[Int]): Tensor[TFloat32] = {

    if (indexes.length > shape(0)) {
      throw new IllegalArgumentException("Indexes cannot be greater than vocabulary size.")
    }

    val tags: Array[String] = Array(SavedModelBundle.DEFAULT_TAG)
    val modelPath = "src/main/resources/embeddings-lookup/"
    val model: SavedModelBundle = TensorflowWrapper.withSafeSavedModelBundleLoader(tags = tags, savedModelDir = modelPath)
    val embeddingsLookup: ConcreteFunction = model.function("embeddings_lookup")

    val tensorResources = new TensorResources()
    val inputs = tensorResources.createTensor(indexes)
    val args: util.Map[String, Tensor[_]] = new util.HashMap()
    args.put("inputs", inputs)
    args.put("embeddings", embeddingsTable)
    val output = embeddingsLookup.call(args)

    val embeddingValues = output.get("output_0").expect(TFloat32.DTYPE)
    embeddingValues
  }

  private def initializeRandomNormal(scope: Scope, seed: Int): Tensor[TFloat32] = {
    val operandShape: Operand[TInt32] = Constant.vectorOf(scope, shape)
    val operandSeed: Operand[TInt32] = Constant.vectorOf(scope, Array[Int](1, seed))
    val statelessRandomNormal = StatelessRandomNormal.create(scope, operandShape, operandSeed)
    statelessRandomNormal.asTensor()
  }

}
