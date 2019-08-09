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

    private final int len;                    // sentence length
    private final int ntypes;                // number of label types
    private final float gammaL;

    FeatureVector[] wordFvs;        // word feature vector

    // word projections U\phi and V\phi
    float[][] wpU;
    float[][] wpV;

    // word projections U2\phi, V2\phi and W2\phi
    float[][] wpU2;
    float[][] wpV2;
    float[][] wpW2;

    private float[][] f;
    private float[][][] labelScores;

    LocalFeatureData(DependencyInstance dependencyInstance,
                     TypedDependencyParser parser) {
        this.dependencyInstance = dependencyInstance;
        pipe = parser.getDependencyPipe();
        synFactory = pipe.getSynFactory();
        options = parser.getOptions();
        parameters = parser.getParameters();

        len = dependencyInstance.getLength();
        ntypes = pipe.getTypes().length;
        int rank = options.rankFirstOrderTensor;
        int rank2 = options.rankSecondOrderTensor;
        gammaL = options.gammaLabel;

        wordFvs = new FeatureVector[len];
        wpU = new float[len][rank];
        wpV = new float[len][rank];

        wpU2 = new float[len][rank2];
        wpV2 = new float[len][rank2];
        wpW2 = new float[len][rank2];


        f = new float[len][ntypes];
        labelScores = new float[len][ntypes][ntypes];

        for (int i = 0; i < len; ++i) {
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

    private void predictLabelsDP(int[] heads, int[] deplbids, boolean addLoss, DependencyArcList arcLis) {

        int lab0 = addLoss ? 0 : 1;

        for (int mod = 1; mod < len; ++mod) {
            int head = heads[mod];
            int dir = head > mod ? 1 : 2;
            int gp = heads[head];
            int pdir = gp > head ? 1 : 2;
            for (int p = lab0; p < ntypes; ++p) {
                if (pipe.getPruneLabel()[dependencyInstance.getXPosTagIds()[head]][dependencyInstance.getXPosTagIds()[mod]][p]) {
                    deplbids[mod] = p;
                    float s1 = 0;
                    if (gammaL > 0)
                        s1 += gammaL * getLabelScoreTheta(heads, deplbids, mod, 1);
                    if (gammaL < 1)
                        s1 += (1 - gammaL) * parameters.dotProductL(wpU[head], wpV[mod], p, dir);
                    for (int q = lab0; q < ntypes; ++q) {
                        float s2 = 0;
                        if (gp != -1) {
                            if (pipe.getPruneLabel()[dependencyInstance.getXPosTagIds()[gp]][dependencyInstance.getXPosTagIds()[head]][q]) {
                                deplbids[head] = q;
                                if (gammaL > 0)
                                    s2 += gammaL * getLabelScoreTheta(heads, deplbids, mod, 2);
                                if (gammaL < 1)
                                    s2 += (1 - gammaL) * parameters.dotProduct2L(wpU2[gp], wpV2[head], wpW2[mod], q, p, pdir, dir);
                            } else s2 = Float.NEGATIVE_INFINITY;
                        }
                        labelScores[mod][p][q] = s1 + s2 + (addLoss && dependencyInstance.getDependencyLabelIds()[mod] != p ? 1.0f : 0.0f);
                    }
                } else Arrays.fill(labelScores[mod][p], Float.NEGATIVE_INFINITY);
            }
        }

        treeDP(0, arcLis, lab0);
        deplbids[0] = dependencyInstance.getDependencyLabelIds()[0];
        getType(0, arcLis, deplbids, lab0);

    }

    private float getLabelScoreTheta(int[] heads, int[] types, int mod, int order) {
        ScoreCollector col = new ScoreCollector(parameters.getParamsL());
        synFactory.createLabelFeatures(col, dependencyInstance, heads, types, mod, order);
        return col.getScore();
    }

    private void treeDP(int i, DependencyArcList arcLis, int lab0) {
        Arrays.fill(f[i], 0);
        int st = arcLis.startIndex(i);
        int ed = arcLis.endIndex(i);
        for (int l = st; l < ed; ++l) {
            int j = arcLis.get(l);
            treeDP(j, arcLis, lab0);
            for (int p = lab0; p < ntypes; ++p) {
                float best = Float.NEGATIVE_INFINITY;
                for (int q = lab0; q < ntypes; ++q) {
                    float s = f[j][q] + labelScores[j][q][p];
                    if (s > best)
                        best = s;
                }
                f[i][p] += best;
            }
        }
    }

    private void getType(int i, DependencyArcList arcLis, int[] types, int lab0) {
        int p = types[i];
        int st = arcLis.startIndex(i);
        int ed = arcLis.endIndex(i);
        for (int l = st; l < ed; ++l) {
            int j = arcLis.get(l);
            int bestq = 0;
            float best = Float.NEGATIVE_INFINITY;
            for (int q = lab0; q < ntypes; ++q) {
                float s = f[j][q] + labelScores[j][q][p];
                if (s > best) {
                    best = s;
                    bestq = q;
                }
            }
            types[j] = bestq;
            getType(j, arcLis, types, lab0);
        }
    }

    void predictLabels(int[] heads, int[] dependencyLabelIds, boolean addLoss) {
        DependencyArcList arcLis = new DependencyArcList(heads);
        predictLabelsDP(heads, dependencyLabelIds, addLoss, arcLis);
    }

}