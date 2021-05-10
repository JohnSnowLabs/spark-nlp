/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
