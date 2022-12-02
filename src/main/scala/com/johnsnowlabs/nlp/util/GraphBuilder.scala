/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.util

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks.{break, breakable}

/** Graph Builder that creates a graph representation as an Adjacency List
  *
  * Adjacency List: An array of lists is used. The size of the array is equal to the number of
  * vertices. Let the array be an array[]. An entry array[i] represents the list of vertices
  * adjacent to the ith vertex.
  * @param numberOfVertices
  */
class GraphBuilder(numberOfVertices: Int) {

  if (numberOfVertices <= 0) {
    throw new IllegalArgumentException("Graph should have at least two vertices")
  }

  private val graph: Map[Int, mutable.Set[Int]] = (0 until numberOfVertices).toList.flatMap {
    vertexIndex =>
      Map(vertexIndex -> mutable.Set[Int]())
  }.toMap

  def addEdge(source: Int, destination: Int): Unit = {
    validateDestinationVertex(destination)
    val adjacentNodes = getAdjacentVertices(source)
    adjacentNodes += destination
  }

  def getNumberOfVertices: Int = {
    graph.size
  }

  def getAdjacentVertices(source: Int): mutable.Set[Int] = {
    val adjacentNodes = graph
      .get(source)
      .orElse(throw new NoSuchElementException(s"Source vertex $source does not exist"))
    adjacentNodes.get
  }

  def edgeExists(source: Int, destination: Int): Boolean = {
    validateDestinationVertex(destination)
    val adjacentNodes = getAdjacentVertices(source)
    adjacentNodes.contains(destination)
  }

  private def validateDestinationVertex(destination: Int): Unit = {
    if (destination > graph.size - 1) {
      throw new NoSuchElementException(s"Destination vertex $destination does not exist")
    }
  }

  /** Find a path using Depth-first search (DFS) algorithm DFS traverses a tree or graph data
    * structures. The algorithm starts at a source node and explores as far as possible along each
    * branch before backtracking It uses a stack to store the path of visited nodes
    */
  def findPath(source: Int, destination: Int): List[Int] = {

    if (source > graph.size || destination > graph.size) {
      throw new IllegalArgumentException(
        "Source or destination vertices cannot be greater than the size of the graph.")
    }

    val visited: Array[Boolean] = (0 until graph.size).toList.map(_ => false).toArray
    val elementsStack: ListBuffer[Int] = ListBuffer(source)
    val pathStack = new ListBuffer[Int]()

    breakable {
      while (elementsStack.nonEmpty) {

        val topVertex: Int = elementsStack.last

        if (!visited(topVertex)) {
          elementsStack.remove(elementsStack.length - 1)
          visited(topVertex) = true
          pathStack += topVertex
          if (pathStack.contains(destination)) {
            break
          }
        }

        val adjacentVertices = getAdjacentVertices(topVertex).iterator

        while (adjacentVertices.hasNext) {
          val vertex = adjacentVertices.next()
          if (!visited(vertex)) {
            elementsStack += vertex
          }
        }

        var cleaningPathStack = true
        while (cleaningPathStack) {
          val topElement = pathStack.last
          val missingVisitedVertices =
            getAdjacentVertices(topElement).filter(vertex => !visited(vertex))
          if (pathStack.length == 1 || missingVisitedVertices.nonEmpty) {
            cleaningPathStack = false
          }
          if (visited(topElement) && missingVisitedVertices.isEmpty) {
            pathStack.remove(pathStack.length - 1)
          }
        }
      }
    }

    pathStack.toList
  }

}
