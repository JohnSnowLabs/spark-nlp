package com.johnsnowlabs.nlp.annotators.parser.dl

import org.scalatest.FlatSpec

class ProjectiveDependencyTreeTest extends FlatSpec {

  "Parser" should "parse heads from a matrix of scores " in {
    val scoresMatrix: Array[Array[Float]] = Array(
      Array(1.3818976f, -0.76371664f, 0.97493255f, -0.16231084f, 1.2365054f, 0.47495997f, 0.27522433f, 2.0086293f),
      Array(0.6893703f, 0.13504124f, 0.7056855f, 0.18809175f, 1.6208357f, 0.9801787f, 0.64658606f, 1.0659467f),
      Array(1.1189872f, 1.3110169f, 1.5882403f, 1.2028598f, 2.7755132f, 1.8355969f, 1.6997403f, 2.0407333f),
      Array(0.5378126f, 1.7041715f, 1.0691303f, 1.3526045f, 3.1520395f, 2.2556906f, 1.995135f, 1.4446837f),
      Array(-1.7294242f, -1.5925231f, -1.2202343f, -1.1268386f, 1.406711f, 0.57233655f, -0.29710543f, -0.4337678f),
      Array(-0.76323766f, -0.87086993f, -0.45489335f, -0.44281876f, 2.228023f, 1.7397033f, 0.83869255f, 0.7721542f),
      Array(0.03176963f, 0.00350606f, 0.2527511f, 0.1749835f, 2.4621768f, 2.081542f, 1.3476499f, 1.3062359f),
      Array(-0.53182995f, -1.4348363f, -0.21265328f, -0.46414566f, 1.7456111f, 1.3787736f, 0.7477962f, 1.5504645f)
    )
    val expectedHeads = List(-1, 2, 0, 2, 3, 3, 3, 2)

    val actualHeads = ProjectiveDependencyTree.parse(scoresMatrix)

    assert(expectedHeads == actualHeads)
  }

}
