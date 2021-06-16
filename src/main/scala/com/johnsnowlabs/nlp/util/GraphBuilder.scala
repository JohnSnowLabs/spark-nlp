package com.johnsnowlabs.nlp.util

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks.{break, breakable}

/**
  * Graph Builder that creates a graph representation as an Adjacency List
  * Adjacency List: An array of lists is used. The size of the array is equal to the number of vertices.
  * Let the array be an array[]. An entry array[i] represents the list of vertices adjacent to the ith vertex.
  * @param numberOfVertices
  */
class GraphBuilder(numberOfVertices: Int) {

  if (numberOfVertices <= 0) {
    throw new IllegalArgumentException("Graph should have at least two vertices")
  }

  private val graph: Map[Int, mutable.Set[Int]] = (0 until numberOfVertices).toList.flatMap{ vertexIndex =>
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
    val adjacentNodes = graph.get(source).orElse(throw new NoSuchElementException(s"Source vertex $source does not exist"))
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

  //TODO: Modify name and add comments that the method implements DFS algorithm
  def depthFirstSearch(source: Int, destination: Int): List[Int]  = {

    val visited: Array[Boolean] = (0 until graph.size).toList.map( _ => false).toArray
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
          val missingVisitedVertices = getAdjacentVertices(topElement).filter(vertex => !visited(vertex))
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
