/*
 * Copyright 2017-2019 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.crf

object VectorMath {

  type Matrix = Array[Array[Float]]
  type Vector = Array[Float]

  def Matrix(n1: Int, n2: Int): Matrix = {
    Array.fill(n1, n2)(0f)
  }

  def I(n: Int): Matrix = {
    val result = Matrix(n, n)
    for (i <- 0 until n)
      result(i)(i) = 1

    result
  }

  def Vector(n: Int, value: Float = 0f): Vector = {
    Array.fill(n)(value)
  }

  def exp(matrix: Matrix): Unit = {
    for (i <- 0 until matrix.length) {
      for (j <- 0 until matrix.length) {
        // ToDo Test work in Doubles
        matrix(i)(j) = Math.exp(matrix(i)(j)).toFloat
      }
    }
  }

  def exp(matrixes: Array[Matrix]): Unit = {
    for (matrix <- matrixes)
      exp(matrix)
  }

  def fillMatrises(matrixes: Array[Matrix], value: Float = 0f): Unit = {
    for (matrix <- matrixes)
      fillMatrix(matrix, value)
  }

  def fillMatrix(matrix: Matrix, value: Float = 0f): Unit = {
    for (vector <- matrix) {
      fillVector(vector, value)
    }
  }

  def fillVector(vector: Vector, value: Float = 0f): Unit = {
    for (i <- 0 until vector.length) {
      vector(i) = value
    }
  }

  def multiply(vector: Vector, a: Float) = {
    for (i <- 0 until vector.length)
      vector(i) *= a
  }

  def copy(from: Matrix, to: Matrix): Unit = {
    for (i <- 0 until from.length)
      copy(from(i), to(i))
  }

  def copy(from: Vector, to: Vector): Unit = {
    require(from.length == to.length)

    for (i <- 0 until from.length)
      to(i) = from(i)
  }

  def mult(a: Matrix, b: Matrix): Matrix = {
    require(a.length > 0)
    require(b.length > 0)

    val result = Matrix(a.length, b(0).length)
    require(a(0).length == b.length)

    for (i <- 0 until a.length) {
      for (j <- 0 until b(0).length) {
        for (l <- 0 until b.length) {
          result(i)(j) += a(i)(l)*b(l)(j)
        }
      }
    }

    result
  }
}

