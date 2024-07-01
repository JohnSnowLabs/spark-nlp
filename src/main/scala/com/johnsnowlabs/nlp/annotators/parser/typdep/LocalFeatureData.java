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

package com.johnsnowlabs.nlp.annotators.parser.typdep;

import com.johnsnowlabs.nlp.annotators.parser.typdep.feature.SyntacticFeatureFactory;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.FeatureVector;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.ScoreCollector;

import java.util.Arrays;

public class LocalFeatureData {

    private DependencyInstance dependencyInstance;
    private DependencyPipe pipe;
    private SyntacticFeatureFactory synFactory;
    private Options options;
    private Parameters parameters;

    private final int sentenceLength;
    private final int numberOfLabelTypes;
    private final float gammaL;

    FeatureVector[] wordFvs;        // word feature vector

    // word projections U\phi and V\phi
    float[][] wpU;
    float[][] wpV;

    // word projections U2\phi, V2\phi and W2\phi
    float[][] wpU2;
    float[][] wpV2;
    float[][] wpW2;

    private float[][] scoresOrProbabilities;
    private float[][][] labelScores;

    private static final int FORWARD_DIRECTION = 1;
    private static final int BACKWARD_DIRECTION = 2;

    LocalFeatureData(DependencyInstance dependencyInstance,
                     TypedDependencyParser parser) {
        this.dependencyInstance = dependencyInstance;
        pipe = parser.getDependencyPipe();
        synFactory = pipe.getSynFactory();
        options = parser.getOptions();
        parameters = parser.getParameters();

        sentenceLength = dependencyInstance.getLength();
        numberOfLabelTypes = pipe.getTypes().length;
        int rank = options.rankFirstOrderTensor;
        int rank2 = options.rankSecondOrderTensor;
        gammaL = options.gammaLabel;

        wordFvs = new FeatureVector[sentenceLength];
        wpU = new float[sentenceLength][rank];
        wpV = new float[sentenceLength][rank];

        wpU2 = new float[sentenceLength][rank2];
        wpV2 = new float[sentenceLength][rank2];
        wpW2 = new float[sentenceLength][rank2];


        scoresOrProbabilities = new float[sentenceLength][numberOfLabelTypes];
        labelScores = new float[sentenceLength][numberOfLabelTypes][numberOfLabelTypes];

        for (int i = 0; i < sentenceLength; ++i) {
            wordFvs[i] = synFactory.createWordFeatures(dependencyInstance, i);

            parameters.projectU(wordFvs[i], wpU[i]);
            parameters.projectV(wordFvs[i], wpV[i]);

            parameters.projectU2(wordFvs[i], wpU2 != null ? wpU2[i] : new float[0]);
            parameters.projectV2(wordFvs[i], wpV2 != null ? wpV2[i] : new float[0]);
            parameters.projectW2(wordFvs[i], wpW2 != null ? wpW2[i] : new float[0]);

        }

    }

    FeatureVector getLabeledFeatureDifference(DependencyInstance gold,
                                              int[] predictedHeads, int[] predictedLabels) {
        FeatureVector dlfv = new FeatureVector();

        int[] actualHeads = gold.getHeads();
        int[] actualLabels = gold.getDependencyLabelIds();

        for (int mod = 1; mod < dependencyInstance.getLength(); ++mod) {
            int head = actualHeads[mod];
            if (actualLabels[mod] != predictedLabels[mod]) {
                dlfv.addEntries(getLabelFeature(actualHeads, actualLabels, mod, 1));
                dlfv.addEntries(getLabelFeature(predictedHeads, predictedLabels, mod, 1), -1.0f);
            }
            if (actualLabels[mod] != predictedLabels[mod] || actualLabels[head] != predictedLabels[head]) {
                dlfv.addEntries(getLabelFeature(actualHeads, actualLabels, mod, 2));
                dlfv.addEntries(getLabelFeature(predictedHeads, predictedLabels, mod, 2), -1.0f);
            }
        }

        return dlfv;
    }

    private FeatureVector getLabelFeature(int[] heads, int[] types, int mod, int order) {
        FeatureVector fv = new FeatureVector();
        synFactory.createLabelFeatures(fv, dependencyInstance, heads, types, mod, order);
        return fv;
    }

    void predictLabels(int[] heads, int[] dependencyLabelIds, boolean addLoss) {
        DependencyArcList arcLis = new DependencyArcList(heads);
        predictLabelsDependencyParser(heads, dependencyLabelIds, addLoss, arcLis);
    }

    private void predictLabelsDependencyParser(int[] heads,
                                               int[] dependencyLabelIds,
                                               boolean addLoss,
                                               DependencyArcList arcList) {
        int startLabelIndex = addLoss ? 0 : 1;
        for (int modifier = 1; modifier < sentenceLength; ++modifier) {
            for (int labelIndex = startLabelIndex; labelIndex < numberOfLabelTypes; ++labelIndex) {
                processLabel(
                    heads,
                    modifier,
                    startLabelIndex,
                    labelIndex,
                    addLoss,
                    dependencyLabelIds);
            }
        }
        finalizePredictions(arcList, startLabelIndex, dependencyLabelIds);
    }

    private int getDirection(int from, int to) {
        return from > to ? FORWARD_DIRECTION : BACKWARD_DIRECTION;
    }

    private void processLabel(
      int[] heads,
      int modifier,
      int startLabelIndex,
      int labelIndex,
      boolean addLoss,
      int[] dependencyLabelIds
    ) {
        int head = heads[modifier];
        if (!shouldPruneLabel(head, modifier, labelIndex)) {
            Arrays.fill(labelScores[modifier][labelIndex], Float.NEGATIVE_INFINITY);
            return;
        }

        dependencyLabelIds[modifier] = labelIndex;
        float score = calculateInitialScore(heads, dependencyLabelIds, modifier, labelIndex);
        for (int index = startLabelIndex; index < numberOfLabelTypes; ++index) {
            float adjustedScore =
                adjustScoreForGrandparent(
                    heads,
                    modifier,
                    labelIndex,
                    index,
                    addLoss,
                    dependencyLabelIds);
            labelScores[modifier][labelIndex][index] = score + adjustedScore;
        }
    }

    private float adjustScoreForGrandparent(
        int[] heads,
        int modifier,
        int labelIndex,
        int currentLabelIndex,
        boolean addLoss,
        int[] dependencyLabelIds
    ) {
        float score = Float.NEGATIVE_INFINITY;

        int head = heads[modifier];
        int grandParent = heads[head];
        int direction = getDirection(head, modifier);
        int parentDirection = getDirection(grandParent, head);

        if (grandParent != -1 && shouldPruneLabel(grandParent, head, currentLabelIndex)) {
            dependencyLabelIds[head] = currentLabelIndex;
            score = 0; // Reset score for new calculations
            if (gammaL > 0)
                score += gammaL * getLabelScoreTheta(heads, dependencyLabelIds, head, BACKWARD_DIRECTION);
            if (gammaL < 1) {
                float dotProduct2L = parameters.dotProduct2L(
                    wpU2[grandParent],
                    wpV2[head],
                    wpW2[modifier],
                    currentLabelIndex,
                    labelIndex,
                    parentDirection,
                    direction);
                score += (1 - gammaL) * dotProduct2L;
            }
            score += (addLoss && dependencyInstance.getDependencyLabelIds()[modifier] != labelIndex ? 1.0f : 0.0f);
        }
        return score;
    }

    private boolean shouldPruneLabel(int dim1, int dim2, int dim3) {
        int[] posTagIds = dependencyInstance.getXPosTagIds();
        return pipe.getPruneLabel()[posTagIds[dim1]][posTagIds[dim2]][dim3];
    }

    private float calculateInitialScore(
        int[] heads,
        int[] dependencyLabelIds,
        int modifier,
        int labelIndex
    ) {
        int head = heads[modifier];
        int direction = getDirection(head, modifier);
        float score = 0;
        if (gammaL > 0)
            score += gammaL * getLabelScoreTheta(heads, dependencyLabelIds, modifier, FORWARD_DIRECTION);
        if (gammaL < 1)
            score += (1 - gammaL) * parameters.dotProductL(wpU[head], wpV[modifier], labelIndex, direction);
        return score;
    }

    private float getLabelScoreTheta(
        int[] heads,
        int[] types,
        int mod,
        int order
    ) {
        ScoreCollector collector = new ScoreCollector(parameters.getParamsL());
        synFactory.createLabelFeatures(collector, dependencyInstance, heads, types, mod, order);
        return collector.getScore();
    }

    private void finalizePredictions(DependencyArcList arcList, int startLabelIndex, int[] dependencyLabelIds) {
        treeDP(0, arcList, startLabelIndex);
        dependencyLabelIds[0] = dependencyInstance.getDependencyLabelIds()[0];
        computeDependencyLabels(0, arcList, dependencyLabelIds, startLabelIndex);
    }

    private void treeDP(int indexNode, DependencyArcList dependencyArcs, int startLabelIndex) {
        Arrays.fill(scoresOrProbabilities[indexNode], 0);
        int startArcIndex = dependencyArcs.startIndex(indexNode);
        int endArcIndex = dependencyArcs.endIndex(indexNode);
        for (int arcIndex = startArcIndex; arcIndex < endArcIndex; ++arcIndex) {
            int currentNode = dependencyArcs.get(arcIndex);
            treeDP(currentNode, dependencyArcs, startLabelIndex);
            for (int labelIndex = startLabelIndex; labelIndex < numberOfLabelTypes; ++labelIndex) {
                float currentScore = scoresOrProbabilities[currentNode][startLabelIndex];
                float currentLabelScore = labelScores[currentNode][startLabelIndex][labelIndex];
                float bestScore = currentScore + currentLabelScore;
                for (int q = startLabelIndex + 1; q < numberOfLabelTypes; ++q) {
                    float score = scoresOrProbabilities[currentNode][q] + labelScores[currentNode][q][labelIndex];
                    if (score > bestScore)
                        bestScore = score;
                }
                scoresOrProbabilities[indexNode][labelIndex] += bestScore;
            }
        }
    }

    private void computeDependencyLabels(int indexNode,
                                         DependencyArcList dependencyArcs,
                                         int[] dependencyLabelIds,
                                         int startLabelIndex) {
        int dependencyLabelId = dependencyLabelIds[indexNode];
        int startArcIndex = dependencyArcs.startIndex(indexNode);
        int endArcIndex = dependencyArcs.endIndex(indexNode);
        for (int arcIndex = startArcIndex; arcIndex < endArcIndex; ++arcIndex) {
            int currentNode = dependencyArcs.get(arcIndex);
            int bestLabel = 0;
            float bestScore = Float.NEGATIVE_INFINITY;
            for (int labelIndex = startLabelIndex; labelIndex < numberOfLabelTypes; ++labelIndex) {
                float currentScore = scoresOrProbabilities[currentNode][labelIndex];
                float currentLabelScore = labelScores[currentNode][labelIndex][dependencyLabelId];
                float totalScore = currentScore + currentLabelScore;
                if (totalScore > bestScore) {
                    bestScore = totalScore;
                    bestLabel = labelIndex;
                }
            }
            if (bestScore == Float.NEGATIVE_INFINITY) {
                // if all scores are -Infinity, assign the original type
                bestLabel = dependencyLabelIds[currentNode];
            }
            dependencyLabelIds[currentNode] = bestLabel;
            computeDependencyLabels(currentNode, dependencyArcs, dependencyLabelIds, startLabelIndex);
        }
    }

}