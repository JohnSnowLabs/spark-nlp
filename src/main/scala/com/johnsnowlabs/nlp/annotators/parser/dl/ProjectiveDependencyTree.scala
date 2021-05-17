package com.johnsnowlabs.nlp.annotators.parser.dl

import scala.collection._

object ProjectiveDependencyTree {

  private case class Tree(maxValue: Float, indexMaxValue: Int)

  /** Parse using Eisner's algorithm */
  def parse(scoresMatrix: Array[Array[Float]]): List[Int] = {
    val rowsSize = scoresMatrix.length
    val columnSize = scoresMatrix(0).length
    if (rowsSize != columnSize) {
      throw new UnsupportedOperationException("Scores must be a squared matrix")
    }

    val (complete, completeBacktrack) = initializeCKYTable(rowsSize)
    val (incomplete, incompleteBacktrack) = initializeCKYTable(rowsSize)
    (1 until rowsSize).toList.foreach { k =>
      (0 until rowsSize - k).toList.foreach { s =>
        val t = s + k

        val (incompleteLeftTree, incompleteRightTree) = createIncompleteItems(s, t, complete, scoresMatrix)
        incomplete(s)(t)(0) = incompleteLeftTree.maxValue
        incompleteBacktrack(s)(t)(0) = s + incompleteLeftTree.indexMaxValue
        incomplete(s)(t)(1) = incompleteRightTree.maxValue
        incompleteBacktrack(s)(t)(1) = s + incompleteRightTree.indexMaxValue

        val (completeLeftTree, completeRightTree) = createCompleteItems(s, t, complete, incomplete)
        complete(s)(t)(0) = completeLeftTree.maxValue
        completeBacktrack(s)(t)(0) = s + completeLeftTree.indexMaxValue
        complete(s)(t)(1) = completeRightTree.maxValue
        completeBacktrack(s)(t)(1) = s + 1 + completeRightTree.indexMaxValue
      }
    }
    val heads = List.fill(rowsSize)(-1).toBuffer
    backtrackEisner(incompleteBacktrack, completeBacktrack, 0 , rowsSize - 1, 1, 1, heads)
    heads.toList
  }

  private def initializeCKYTable(numberOfWords: Int): (Array[Array[Array[Float]]], Array[Array[Array[Int]]]) = {
    val zerosMatrix = Array.ofDim[Float](numberOfWords, numberOfWords, 2)
    val negativeOnesMatrix = zerosMatrix.map(secondDim => secondDim.map(thirdDim =>
      thirdDim.map(element => element.toInt - 1)))
    (zerosMatrix, negativeOnesMatrix)
  }

  private def createIncompleteItems(s: Int, t: Int, complete: Array[Array[Array[Float]]], scoresMatrix: Array[Array[Float]]) = {
    val elementsSecondDimension = (s until t).map(i => complete(s)(i)(1)).toList
    val elementsFirstDimension = ((s + 1) until (t + 1)).map( i => complete(i)(t)(0)).toList
    val sumElements = (elementsSecondDimension, elementsFirstDimension).zipped.map(_ + _)

    var scoreValue = scoresMatrix(t)(s) + 1
    val leftTreeValues = sumElements.map(_ + scoreValue).zipWithIndex.maxBy(_._1)

    scoreValue = scoresMatrix(s)(t) + 1
    val rightTreeValues = sumElements.map(_ + scoreValue).zipWithIndex.maxBy(_._1)

    (Tree(leftTreeValues._1, leftTreeValues._2), Tree(rightTreeValues._1, rightTreeValues._2))
  }

  private def createCompleteItems(s: Int, t: Int, complete: Array[Array[Array[Float]]],
                                  incomplete: Array[Array[Array[Float]]]) = {
    val completeElementsSecondDimension = (s until t).map(i => complete(s)(i)(0)).toList
    val incompleteElementsFirstDimension = (s until t).map(i => incomplete(i)(t)(0)).toList
    val leftTreeValues = (completeElementsSecondDimension, incompleteElementsFirstDimension)
      .zipped.map(_ + _).zipWithIndex.maxBy(_._1)

    val incompleteElementsSecondDimension = ((s + 1) until (t + 1)).map( i => incomplete(s)(i)(1)).toList
    val completeElementsFirstDimension = ((s + 1) until (t + 1)).map( i => complete(i)(t)(1)).toList
    val rightTreeValues = (incompleteElementsSecondDimension, completeElementsFirstDimension)
      .zipped.map(_ + _).zipWithIndex.maxBy(_._1)

    (Tree(leftTreeValues._1, leftTreeValues._2), Tree(rightTreeValues._1, rightTreeValues._2))
  }

  /** Backtracking step in Eisner's algorithm.
    * @param incompleteBacktrack is a (numberOfWords)-by-(numberOfWords) array indexed by a start position,
    * an end position, and a direction flag (0 means left, 1 means right). This array contains the arg-maxes of
    * each step in the Eisner algorithm when building *incomplete* spans.
    * @param completeBacktrack is a (numberOfWords)-by-(numberOfWords) array indexed by a start position,
    * an end position, and a direction flag (0 means left, 1 means right). This array contains the arg-maxes of
    * each step in the Eisner algorithm when building *complete* spans.
    * @param s is the current start of the span
    * @param t is the current end of the span
    * @param direction is 0 (left attachment) or 1 (right attachment)
    * @param complete is 1 if the current span is complete, and 0 otherwise
    * @param heads is a (numberOfWords)-sized array of integers which is a placeholder for storing the head of each word.
    * */
  private def backtrackEisner(incompleteBacktrack: Array[Array[Array[Int]]],
                              completeBacktrack: Array[Array[Array[Int]]],
                              s: Int, t: Int, direction: Int, complete: Int, heads: mutable.Buffer[Int]): Option[Int] = {
    if (s == t) {
      None
    } else {
      if (complete == 1) {
        val r = completeBacktrack(s)(t)(direction)
        if (direction == 0) {
          backtrackEisner(incompleteBacktrack, completeBacktrack, s, r, 0, 1, heads)
          backtrackEisner(incompleteBacktrack, completeBacktrack, r, t, 0, 0, heads)
          Some(r)
        } else {
          backtrackEisner(incompleteBacktrack, completeBacktrack, s, r, 1, 0, heads)
          backtrackEisner(incompleteBacktrack, completeBacktrack, r, t, 1, 1, heads)
          Some(r)
        }
      } else {
        val r = incompleteBacktrack(s)(t)(direction)
        if (direction == 0) {
          heads(s) = t
          backtrackEisner(incompleteBacktrack, completeBacktrack, s, r, 1, 1, heads)
          backtrackEisner(incompleteBacktrack, completeBacktrack, r + 1, t, 0, 1, heads)
          Some(r)
        } else {
          heads(t) = s
          backtrackEisner(incompleteBacktrack, completeBacktrack, s, r, 1, 1, heads)
          backtrackEisner(incompleteBacktrack, completeBacktrack, r + 1, t, 0, 1, heads)
          Some(r)
        }
      }
    }
  }

}