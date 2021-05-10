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

import org.scalatest.FlatSpec
import org.tensorflow.Tensor
import org.tensorflow.types.TFloat32

class TensorflowEmbeddingLookupTest extends FlatSpec with EagerSessionBuilder {

  val tensorResources = new TensorResources()

  "Tensorflow Embeddings Lookup" should "create embedding table of required size" in {
    val embeddingsSize = 100
    val vocabularySize = 150
    val expectedEmbeddingTableSize = Array(vocabularySize, embeddingsSize)

    val embeddings = new TensorflowEmbeddingLookup(embeddingsSize, vocabularySize, scope)

    assert(embeddings.shape sameElements expectedEmbeddingTableSize)

  }

  it should "initialize same embeddings values for the same seed" in {
    val embeddingsSize = 3
    val vocabularySize = 1
    val seed = 4
    val expectedEmbeddingsValues = Array(1.0017798f, 2.329537f, -0.60242355f)
    val expectedEmbeddingTable = tensorResources.createTensor(Array(expectedEmbeddingsValues))
    val embeddings = new TensorflowEmbeddingLookup(embeddingsSize, vocabularySize, scope)

    val actualEmbeddingsTable = embeddings.initializeTable(seed = seed)

    assert(expectedEmbeddingTable.shape().asArray() sameElements actualEmbeddingsTable.shape().asArray())
    val actualEmbeddingsValues = TensorResources.extractFloats(expectedEmbeddingTable)
    assert(expectedEmbeddingsValues sameElements actualEmbeddingsValues)
  }

  it should "rise an error when initializer argument get unexpected value" in {
    val embeddingsSize = 3
    val vocabularySize = 1

    val embeddings = new TensorflowEmbeddingLookup(embeddingsSize, vocabularySize, scope)

    val errorMessage = intercept[IllegalArgumentException] {
      embeddings.initializeTable("someInitializer")
    }

    assert(errorMessage.getMessage == "Undefined initializer.")

  }

  it should "lookup an embedding based on index values" in {
    val indexes = Array(1, 3)
    val row0 = Array(1.01f, 2.32f, -0.60f)
    val row1 = Array(1.22f, -5.32f, -8.24f)
    val row2 = Array(3.25f, 7.82f, 3.4f)
    val row3 = Array(1.12f, -9.32f, -7.34f)
    val mockEmbeddingsValues = Array(row0, row1, row2, row3)
    val mockEmbeddingsTable = tensorResources.createTensor(mockEmbeddingsValues).asInstanceOf[Tensor[TFloat32]]
    val embeddings = new TensorflowEmbeddingLookup(3, 4, scope)
    val expectedValues = row1 ++ row3

    val actualValues = embeddings.lookup(mockEmbeddingsTable, indexes)

    assert(expectedValues sameElements TensorResources.extractFloats(actualValues))

  }

  it should "raise error when indexes to lookup are not consistent with vocabulary size" in {
    val indexes = Array(1, 2, 3)
    val mockEmbeddingsValues = Array(Array(1.01f, 2.32f, -0.60f))
    val mockEmbeddingsTable = tensorResources.createTensor(mockEmbeddingsValues).asInstanceOf[Tensor[TFloat32]]
    val embeddings = new TensorflowEmbeddingLookup(3, 1, scope)

    val errorMessage = intercept[IllegalArgumentException] {
      embeddings.lookup(mockEmbeddingsTable, indexes)
    }

    assert(errorMessage.getMessage == "Indexes cannot be greater than vocabulary size.")

  }

}
