package com.johnsnowlabs.nlp.annotators.parser.typdep;

import com.johnsnowlabs.nlp.annotators.parser.typdep.util.FeatureVector;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.Utils;

import java.io.Serializable;

public class Parameters implements Serializable {

    private static final long serialVersionUID = 1L;

    private float regularization;
    private float gammaLabel;
    private int rankFirstOrderTensor;
    private int rankSecondOrderTensor;

    public void setGammaLabel(float gammaLabel) {
        this.gammaLabel = gammaLabel;
    }

    public void setRankFirstOrderTensor(int rankFirstOrderTensor) {
        this.rankFirstOrderTensor = rankFirstOrderTensor;
    }

    public void setRankSecondOrderTensor(int rankSecondOrderTensor) {
        this.rankSecondOrderTensor = rankSecondOrderTensor;
    }

    private int numberWordFeatures;
    private int T;
    private int DL;

    public int getT() {
        return T;
    }

    public int getNumberWordFeatures() {
        return numberWordFeatures;
    }

    public int getDL() {
        return DL;
    }

    private float[] paramsL;

    public float[] getParamsL() {
        return paramsL;
    }

    private transient float[] totalL;

    private float[][] U;
    private float[][] V;
    private float[][] WL;

    public float[][] getU() {
        return U;
    }

    public float[][] getV() {
        return V;
    }

    public float[][] getWL() {
        return WL;
    }

    private float[][] U2;
    private float[][] V2;
    private float[][] W2;
    private float[][] X2L;

    public float[][] getU2() {
        return U2;
    }

    public float[][] getV2() {
        return V2;
    }

    public float[][] getW2() {
        return W2;
    }

    public float[][] getX2L() {
        return X2L;
    }

    public float[][] getY2L() {
        return Y2L;
    }

    private float[][] Y2L;


    private float[][] totalU;
    private float[][] totalV;
    private float[][] totalWL;
    private float[][] totalU2;
    private float[][] totalV2;
    private float[][] totalW2;
    private float[][] totalX2L;
    private float[][] totalY2L;

    private FeatureVector[] dU;
    private FeatureVector[] dV;
    private FeatureVector[] dWL;
    private FeatureVector[] dU2;
    private FeatureVector[] dV2;
    private FeatureVector[] dW2;
    private FeatureVector[] dX2L;
    private FeatureVector[] dY2L;

    public Parameters(DependencyPipe pipe, Options options)
    {
        numberWordFeatures = pipe.getSynFactory().getNumberWordFeatures();
        T = pipe.getTypes().length;
        DL = T * 3;
        regularization = options.regularization;
        gammaLabel = options.gammaLabel;
        rankFirstOrderTensor = options.rankFirstOrderTensor;
        rankSecondOrderTensor = options.rankSecondOrderTensor;

        int sizeL = pipe.getSynFactory().getNumberLabeledArcFeatures() + 1;
        paramsL = new float[sizeL];
        totalL = new float[sizeL];

        U = new float[numberWordFeatures][rankFirstOrderTensor];
        V = new float[numberWordFeatures][rankFirstOrderTensor];
        WL = new float[DL][rankFirstOrderTensor];
        totalU = new float[numberWordFeatures][rankFirstOrderTensor];
        totalV = new float[numberWordFeatures][rankFirstOrderTensor];
        totalWL = new float[DL][rankFirstOrderTensor];
        dU = new FeatureVector[rankFirstOrderTensor];
        dV = new FeatureVector[rankFirstOrderTensor];
        dWL = new FeatureVector[rankFirstOrderTensor];

        U2 = new float[numberWordFeatures][rankSecondOrderTensor];
        V2 = new float[numberWordFeatures][rankSecondOrderTensor];
        W2 = new float[numberWordFeatures][rankSecondOrderTensor];
        X2L = new float[DL][rankSecondOrderTensor];
        Y2L = new float[DL][rankSecondOrderTensor];
        totalU2 = new float[numberWordFeatures][rankSecondOrderTensor];
        totalV2 = new float[numberWordFeatures][rankSecondOrderTensor];
        totalW2 = new float[numberWordFeatures][rankSecondOrderTensor];
        totalX2L = new float[DL][rankSecondOrderTensor];
        totalY2L = new float[DL][rankSecondOrderTensor];
        dU2 = new FeatureVector[rankSecondOrderTensor];
        dV2 = new FeatureVector[rankSecondOrderTensor];
        dW2 = new FeatureVector[rankSecondOrderTensor];
        dX2L = new FeatureVector[rankSecondOrderTensor];
        dY2L = new FeatureVector[rankSecondOrderTensor];

    }

    private void copyValue(float[][] target, float[][] source)
    {
        int n = source.length;
        for (int i = 0; i < n; ++i)
            target[i] = source[i].clone();
    }

    public void assignTotal()
    {
        copyValue(totalU, U);
        copyValue(totalV, V);
        copyValue(totalWL, WL);

        copyValue(totalU2, U2);
        copyValue(totalV2, V2);
        copyValue(totalW2, W2);
        copyValue(totalX2L, X2L);
        copyValue(totalY2L, Y2L);

    }

    private void assignColumn(float[][] mat, int col, float[] values)
    {
        for (int id = 0, tot=values.length; id < tot; ++id)
            mat[id][col] = values[id];
    }

    public void randomlyInit()
    {

        for (int i = 0; i < rankFirstOrderTensor; ++i) {
            assignColumn(U, i, Utils.getRandomUnitVector(numberWordFeatures));
            assignColumn(V, i, Utils.getRandomUnitVector(numberWordFeatures));
            assignColumn(WL, i, Utils.getRandomUnitVector(DL));
        }

        for (int i = 0; i < rankSecondOrderTensor; ++i) {
            assignColumn(U2, i, Utils.getRandomUnitVector(numberWordFeatures));
            assignColumn(V2, i, Utils.getRandomUnitVector(numberWordFeatures));
            assignColumn(W2, i, Utils.getRandomUnitVector(numberWordFeatures));
            assignColumn(X2L, i, Utils.getRandomUnitVector(DL));
            assignColumn(Y2L, i, Utils.getRandomUnitVector(DL));
        }

        assignTotal();
    }

    private void averageTheta(float[] a, float[] totala, int T, float c)
    {
        int n = a.length;
        for (int i = 0; i < n; ++i) {
            a[i] += c*totala[i]/T;
        }
    }

    private void averageTensor(float[][] a, float[][] totala, int T, float c)
    {
        int n = a.length;
        if (n == 0)
            return;
        int m = a[0].length;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j) {
                a[i][j] += c*totala[i][j]/T;
            }
    }

    public void averageParameters(int T, float c)
    {
        averageTheta(paramsL, totalL, T, c);

        averageTensor(U, totalU, T, c);
        averageTensor(V, totalV, T, c);
        averageTensor(WL, totalWL, T, c);


        averageTensor(U2, totalU2, T, c);
        averageTensor(V2, totalV2, T, c);
        averageTensor(W2, totalW2, T, c);
        averageTensor(X2L, totalX2L, T, c);
        averageTensor(Y2L, totalY2L, T, c);

    }

    private void printStat(float[][] a, String s)
    {
        int n = a.length;
        float sum = 0;
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; ++i) {
            sum += Utils.squaredSum(a[i]);
            min = Math.min(min, Utils.min(a[i]));
            max = Math.max(max, Utils.max(a[i]));
        }
        System.out.printf(" |%s|^2: %f min: %f\tmax: %f%n", s, sum, min, max);
    }

    public void printStat()
    {
        printStat(U, "U");
        printStat(V, "V");
        printStat(WL, "WL");


        printStat(U2, "U2");
        printStat(V2, "V2");
        printStat(W2, "W2");
        printStat(X2L, "X2L");
        printStat(Y2L, "Y2L");

    }

    private void projectMat(float[][] mat, FeatureVector fv, float[] proj)
    {
        int rank = proj.length;
        for (int id = 0, n = fv.size(); id < n; ++id) {
            int i = fv.x(id);
            float w = fv.value(id);
            for (int j = 0; j < rank; ++j)
                proj[j] += mat[i][j] * w;
        }
    }

    public void projectU(FeatureVector fv, float[] proj)
    {
        projectMat(U, fv, proj);
    }

    public void projectV(FeatureVector fv, float[] proj)
    {
        projectMat(V, fv, proj);
    }

    public void projectU2(FeatureVector fv, float[] proj)
    {
        projectMat(U2, fv, proj);
    }

    public void projectV2(FeatureVector fv, float[] proj)
    {
        projectMat(V2, fv, proj);
    }

    public void projectW2(FeatureVector fv, float[] proj)
    {
        projectMat(W2, fv, proj);
    }

    public float dotProductL(float[] proju, float[] projv, int lab, int dir)
    {
        float sum = 0;
        for (int r = 0; r < rankFirstOrderTensor; ++r)
            sum += proju[r] * projv[r] * (WL[lab][r] + WL[dir*T+lab][r]);
        return sum;
    }

    public float dotProduct2L(float[] proju, float[] projv, float[] projw,
                              int plab, int lab, int pdir, int dir)
    {
        float sum = 0;
        for (int r = 0; r < rankSecondOrderTensor; ++r)
            sum += proju[r] * projv[r] * projw[r] * (X2L[plab][r] + X2L[pdir*T+plab][r])
                    * (Y2L[lab][r] + Y2L[dir*T+lab][r]);
        return sum;
    }

    private void addTheta(float[] a, float[] totala, FeatureVector da,
                          float coeff, float coeff2)
    {
        if (da == null)
            return;
        for (int i = 0, K = da.size(); i < K; ++i) {
            int x = da.x(i);
            float z = da.value(i);
            a[x] += coeff * z;
            totala[x] += coeff2 * z;
        }
    }

    private void addTensor(float[][] a, float[][] totala, FeatureVector[] da,
                           float coeff, float coeff2)
    {
        int n = da.length;
        for (int k = 0; k < n; ++k) {
            FeatureVector dak = da[k];
            if (dak == null)
                continue;
            for (int i = 0, K = dak.size(); i < K; ++i) {
                int x = dak.x(i);
                float z = dak.value(i);
                a[x][k] += coeff * z;
                totala[x][k] += coeff2 * z;
            }
        }
    }

    public float updateLabel(DependencyInstance gold, int[] predictedHeads, int[] predictedLabels,
                             LocalFeatureData localFeatureData, int updCnt)
    {
        int[] actDeps = gold.getHeads();
        int[] actLabs = gold.getDependencyLabelIds();

        float labelDistance = getLabelDistance(actLabs, predictedLabels);

        FeatureVector labeledFeatureDifference = localFeatureData.getLabeledFeatureDifference(gold, predictedHeads, predictedLabels);
        float loss = - labeledFeatureDifference.dotProduct(paramsL)* gammaLabel + labelDistance;
        float l2norm = labeledFeatureDifference.squaredL2NormUnsafe() * gammaLabel * gammaLabel;

        // update U
        for (int k = 0; k < rankFirstOrderTensor; ++k) {
            FeatureVector dUk = getdUL(k, localFeatureData, actDeps, actLabs, predictedHeads, predictedLabels);
            l2norm += dUk.squaredL2NormUnsafe() * (1- gammaLabel) * (1- gammaLabel);
            for (int u = 0, n = dUk.size(); u < n; ++u)
                loss -= U[dUk.x(u)][k] * dUk.value(u) * (1- gammaLabel);
            dU[k] = dUk;
        }
        // update V
        for (int k = 0; k < rankFirstOrderTensor; ++k) {
            FeatureVector dVk = getdVL(k, localFeatureData, actDeps, actLabs, predictedHeads, predictedLabels);
            l2norm += dVk.squaredL2NormUnsafe() * (1- gammaLabel) * (1- gammaLabel);
            dV[k] = dVk;
        }
        // update WL
        for (int k = 0; k < rankFirstOrderTensor; ++k) {
            FeatureVector dWLk = getdWL(k, localFeatureData, actDeps, actLabs, predictedHeads, predictedLabels);
            l2norm += dWLk.squaredL2NormUnsafe() * (1- gammaLabel) * (1- gammaLabel);
            dWL[k] = dWLk;
        }

        // update U2
        for (int k = 0; k < rankSecondOrderTensor; ++k) {
            FeatureVector dU2k = getdU2L(k, localFeatureData, actDeps, actLabs, predictedLabels);
            l2norm += dU2k.squaredL2NormUnsafe() * (1 - gammaLabel) * (1 - gammaLabel);
            for (int u = 0, n = dU2k.size(); u < n; ++u)
                loss -= U2[dU2k.x(u)][k] * dU2k.value(u) * (1 - gammaLabel);
            dU2[k] = dU2k;
        }
        // update V2
        for (int k = 0; k < rankSecondOrderTensor; ++k) {
            FeatureVector dV2k = getdV2L(k, localFeatureData, actDeps, actLabs, predictedLabels);
            l2norm += dV2k.squaredL2NormUnsafe() * (1 - gammaLabel) * (1 - gammaLabel);
            dV2[k] = dV2k;
        }
        // update W2
        for (int k = 0; k < rankSecondOrderTensor; ++k) {
            FeatureVector dW2k = getdW2L(k, localFeatureData, actDeps, actLabs, predictedLabels);
            l2norm += dW2k.squaredL2NormUnsafe() * (1 - gammaLabel) * (1 - gammaLabel);
            dW2[k] = dW2k;
        }
        // update X2L
        for (int k = 0; k < rankSecondOrderTensor; ++k) {
            FeatureVector dX2Lk = getdX2L(k, localFeatureData, actDeps, actLabs, predictedLabels);
            l2norm += dX2Lk.squaredL2NormUnsafe() * (1 - gammaLabel) * (1 - gammaLabel);
            dX2L[k] = dX2Lk;
        }
        // update Y2L
        for (int k = 0; k < rankSecondOrderTensor; ++k) {
            FeatureVector dY2Lk = getdY2L(k, localFeatureData, actDeps, actLabs, predictedLabels);
            l2norm += dY2Lk.squaredL2NormUnsafe() * (1 - gammaLabel) * (1 - gammaLabel);
            dY2L[k] = dY2Lk;
        }

        float alpha = loss/l2norm;
        alpha = Math.min(regularization, alpha);
        if (alpha > 0) {
            float coeff;
            float coeff2;

            coeff = alpha * gammaLabel;
            coeff2 = coeff * (1-updCnt);
            addTheta(paramsL, totalL, labeledFeatureDifference, coeff, coeff2);

            coeff = alpha * (1- gammaLabel);
            coeff2 = coeff * (1-updCnt);
            addTensor(U, totalU, dU, coeff, coeff2);
            addTensor(V, totalV, dV, coeff, coeff2);
            addTensor(WL, totalWL, dWL, coeff, coeff2);

            addTensor(U2, totalU2, dU2, coeff, coeff2);
            addTensor(V2, totalV2, dV2, coeff, coeff2);
            addTensor(W2, totalW2, dW2, coeff, coeff2);
            addTensor(X2L, totalX2L, dX2L, coeff, coeff2);
            addTensor(Y2L, totalY2L, dY2L, coeff, coeff2);

        }
        return loss;
    }

    private float getLabelDistance(int[] actualLabels, int[] predictedLabels)
    {
        float distance = 0;
        for (int i = 1; i < actualLabels.length; ++i) {
            if (actualLabels[i] != predictedLabels[i]) distance += 1;
        }
        return distance;
    }

    private FeatureVector getdUL(int k, LocalFeatureData lfd, int[] actualHeads, int[] actualLabels,
                                 int[] predDeps, int[] predLabs) {
        float[][] wpV = lfd.wpV;
        FeatureVector[] wordFvs = lfd.wordFvs;
        int L = wordFvs.length;
        FeatureVector dU = new FeatureVector();
        for (int mod = 1; mod < L; ++mod) {
            assert(actualHeads[mod] == predDeps[mod]);
            int head  = actualHeads[mod];
            int dir = head > mod ? 1 : 2;
            int lab  = actualLabels[mod];
            int lab2 = predLabs[mod];
            if (lab == lab2){
                continue;
            }
            float dotv = wpV[mod][k];
            dU.addEntries(wordFvs[head], dotv * (WL[lab][k] + WL[dir*T+lab][k])
                    - dotv * (WL[lab2][k] + WL[dir*T+lab2][k]));
        }
        return dU;
    }

    private FeatureVector getdVL(int k, LocalFeatureData localFeatureData, int[] actualHeads, int[] actualLabels,
                                 int[] predDeps, int[] predLabs) {
        float[][] wpU = localFeatureData.wpU;
        FeatureVector[] wordFvs = localFeatureData.wordFvs;
        int L = wordFvs.length;
        FeatureVector dV = new FeatureVector();
        for (int mod = 1; mod < L; ++mod) {
            assert(actualHeads[mod] == predDeps[mod]);
            int head  = actualHeads[mod];
            int dir = head > mod ? 1 : 2;
            int lab  = actualLabels[mod];
            int lab2 = predLabs[mod];
            if (lab == lab2) continue;
            float dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
            dV.addEntries(wordFvs[mod], dotu  * (WL[lab][k] + WL[dir*T+lab][k])
                    - dotu * (WL[lab2][k] + WL[dir*T+lab2][k]));
        }
        return dV;
    }

    private FeatureVector getdWL(int k, LocalFeatureData lfd, int[] actualHeads, int[] actualLabels,
                                 int[] predDeps, int[] predLabs) {
        float[][] wpU = lfd.wpU, wpV = lfd.wpV;
        FeatureVector[] wordFvs = lfd.wordFvs;
        int L = wordFvs.length;
        float[] dWL = new float[DL];
        for (int mod = 1; mod < L; ++mod) {
            assert(actualHeads[mod] == predDeps[mod]);
            int head = actualHeads[mod];
            int dir = head > mod ? 1 : 2;
            int lab  = actualLabels[mod];
            int lab2 = predLabs[mod];
            if (lab == lab2) continue;
            float dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
            float dotv = wpV[mod][k];  //wordFvs[mod].dotProduct(V[k]);
            dWL[lab] += dotu * dotv;
            dWL[dir*T+lab] += dotu * dotv;
            dWL[lab2] -= dotu * dotv;
            dWL[dir*T+lab2] -= dotu * dotv;
        }

        FeatureVector dWLfv = new FeatureVector();
        for (int i = 0; i < DL; ++i)
            dWLfv.addEntry(i, dWL[i]);
        return dWLfv;
    }

    private FeatureVector getdU2L(int k, LocalFeatureData localFeatureData, int[] actualHeads,
                                  int[] actualLabels, int[] predLabs) {
        float[][] wpV2 = localFeatureData.wpV2;
        float[][] wpW2 = localFeatureData.wpW2;
        FeatureVector[] wordFvs = localFeatureData.wordFvs;
        int L = wordFvs.length;
        FeatureVector dU2 = new FeatureVector();
        for (int mod = 1; mod < L; ++mod) {
            int head  = actualHeads[mod];
            int gp = actualHeads[head];
            if (gp == -1)
                continue;
            int dir = head > mod ? 1 : 2;
            int pdir = gp > head ? 1 : 2;
            int lab  = actualLabels[mod];
            int lab2 = predLabs[mod];
            int plab = actualLabels[head];
            int plab2 = predLabs[head];
            if (lab == lab2 && plab == plab2) continue;
            float dotv2 = wpV2[head][k];
            float dotw2 = wpW2[mod][k];
            dU2.addEntries(wordFvs[gp], dotv2 * dotw2 * (X2L[plab][k] + X2L[pdir*T+plab][k]) * (Y2L[lab][k] + Y2L[dir*T+lab][k])
                    - dotv2 * dotw2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]) * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]));
        }
        return dU2;
    }

    private FeatureVector getdV2L(int k, LocalFeatureData localFeatureData, int[] actualHeads,
                                  int[] actualLabels, int[] predictedLabels) {
        float[][] wpU2 = localFeatureData.wpU2;
        float[][] wpW2 = localFeatureData.wpW2;
        FeatureVector[] wordFvs = localFeatureData.wordFvs;
        int L = wordFvs.length;
        FeatureVector dV2 = new FeatureVector();
        for (int mod = 1; mod < L; ++mod) {
            int head  = actualHeads[mod];
            int gp = actualHeads[head];
            if (gp == -1)
                continue;
            int dir = head > mod ? 1 : 2;
            int pdir = gp > head ? 1 : 2;
            int lab  = actualLabels[mod];
            int lab2 = predictedLabels[mod];
            int plab = actualLabels[head];
            int plab2 = predictedLabels[head];
            if (lab == lab2 && plab == plab2) continue;
            float dotu2 = wpU2[gp][k];
            float dotw2 = wpW2[mod][k];
            dV2.addEntries(wordFvs[head], dotu2 * dotw2 * (X2L[plab][k] + X2L[pdir*T+plab][k]) * (Y2L[lab][k] + Y2L[dir*T+lab][k])
                    - dotu2 * dotw2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]) * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]));
        }
        return dV2;
    }

    private FeatureVector getdW2L(int k, LocalFeatureData localFeatureData, int[] actualHeads,
                                  int[] actualLabels, int[] predictedLabels) {
        float[][] wpU2 = localFeatureData.wpU2;
        float[][] wpV2 = localFeatureData.wpV2;
        FeatureVector[] wordFvs = localFeatureData.wordFvs;
        int L = wordFvs.length;
        FeatureVector dW2 = new FeatureVector();
        for (int mod = 1; mod < L; ++mod) {
            int head  = actualHeads[mod];
            int gp = actualHeads[head];
            if (gp == -1)
                continue;
            int dir = head > mod ? 1 : 2;
            int pdir = gp > head ? 1 : 2;
            int lab  = actualLabels[mod];
            int lab2 = predictedLabels[mod];
            int plab = actualLabels[head];
            int plab2 = predictedLabels[head];
            if (lab == lab2 && plab == plab2) continue;
            float dotu2 = wpU2[gp][k];
            float dotv2 = wpV2[head][k];
            dW2.addEntries(wordFvs[mod], dotu2 * dotv2 * (X2L[plab][k] + X2L[pdir*T+plab][k]) * (Y2L[lab][k] + Y2L[dir*T+lab][k])
                    - dotu2 * dotv2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]) * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]));
        }
        return dW2;
    }

    private FeatureVector getdX2L(int k, LocalFeatureData localFeatureData, int[] acutalHeads,
                                  int[] actualLabels, int[] predictedLabels) {
        float[][] wpU2 = localFeatureData.wpU2, wpV2 = localFeatureData.wpV2, wpW2 = localFeatureData.wpW2;
        FeatureVector[] wordFvs = localFeatureData.wordFvs;
        int L = wordFvs.length;
        float[] dX2L = new float[DL];
        for (int mod = 1; mod < L; ++mod) {
            int head  = acutalHeads[mod];
            int gp = acutalHeads[head];
            if (gp == -1)
                continue;
            int dir = head > mod ? 1 : 2;
            int pdir = gp > head ? 1 : 2;
            int lab  = actualLabels[mod];
            int lab2 = predictedLabels[mod];
            int plab = actualLabels[head];
            int plab2 = predictedLabels[head];
            if (lab == lab2 && plab == plab2) continue;
            float dotu2 = wpU2[gp][k];
            float dotv2 = wpV2[head][k];
            float dotw2 = wpW2[mod][k];
            float val = dotu2 * dotv2 * dotw2 * (Y2L[lab][k] + Y2L[dir*T+lab][k]);
            float val2 = dotu2 * dotv2 * dotw2 * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]);
            dX2L[plab] += val;
            dX2L[pdir*T+plab] += val;
            dX2L[plab2] -= val2;
            dX2L[pdir*T+plab2] -= val2;
        }

        FeatureVector dX2Lfv = new FeatureVector();
        for (int i = 0; i < DL; ++i)
            dX2Lfv.addEntry(i, dX2L[i]);
        return dX2Lfv;
    }

    private FeatureVector getdY2L(int k, LocalFeatureData localFeatureData, int[] actualHeads,
                                  int[] actualLabels, int[] predictedLabels) {
        float[][] wpU2 = localFeatureData.wpU2, wpV2 = localFeatureData.wpV2, wpW2 = localFeatureData.wpW2;
        FeatureVector[] wordFvs = localFeatureData.wordFvs;
        int L = wordFvs.length;
        float[] dY2L = new float[DL];
        for (int mod = 1; mod < L; ++mod) {
            int head  = actualHeads[mod];
            int gp = actualHeads[head];
            if (gp == -1)
                continue;
            int dir = head > mod ? 1 : 2;
            int pdir = gp > head ? 1 : 2;
            int lab  = actualLabels[mod];
            int lab2 = predictedLabels[mod];
            int plab = actualLabels[head];
            int plab2 = predictedLabels[head];
            if (lab == lab2 && plab == plab2) continue;
            float dotu2 = wpU2[gp][k];
            float dotv2 = wpV2[head][k];
            float dotw2 = wpW2[mod][k];
            float val = dotu2 * dotv2 * dotw2 * (X2L[plab][k] + X2L[pdir*T+plab][k]);
            float val2 = dotu2 * dotv2 * dotw2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]);
            dY2L[lab] += val;
            dY2L[dir*T+lab] += val;
            dY2L[lab2] -= val2;
            dY2L[dir*T+lab2] -= val2;
        }

        FeatureVector dY2Lfv = new FeatureVector();
        for (int i = 0; i < DL; ++i)
            dY2Lfv.addEntry(i, dY2L[i]);
        return dY2Lfv;
    }

}
