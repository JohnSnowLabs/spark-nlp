package com.johnsnowlabs.ml.tensorflow

import org.tensorflow.Operand
import org.tensorflow.op.Scope
import org.tensorflow.op.math.{AccumulateN, AddN}
import org.tensorflow.types.{TFloat32, TInt32}
import org.tensorflow.types.family.TNumber

import scala.collection.JavaConverters._

object TensorMathResources {

  def sumTensors[T <: TNumber](scope: Scope, tensors: Seq[Operand[T]]): Operand[T] = {
    val elementWiseSum = tensors.head.data() match {
      case float: TFloat32 => AddN.create(scope, tensors.asInstanceOf[Seq[Operand[TFloat32]]].toList.asJava)
      case int: TInt32 => AddN.create(scope, tensors.asInstanceOf[Seq[Operand[TInt32]]].toList.asJava)
    }
    elementWiseSum.asInstanceOf[Operand[T]]
  }

}
