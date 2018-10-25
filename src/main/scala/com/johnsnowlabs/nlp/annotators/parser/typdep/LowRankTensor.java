package com.johnsnowlabs.nlp.annotators.parser.typdep;

import com.johnsnowlabs.nlp.annotators.parser.typdep.util.Utils;

import java.util.ArrayList;

public class LowRankTensor {

    private int dim;
    private int rank;
    private int[] N;
    private ArrayList<MatEntry> list;

    private static final int MAX_ITER=1000;

    LowRankTensor(int[] N, int rank)
    {
        this.N = N.clone();
        dim = N.length;
        this.rank = rank;
        list = new ArrayList<>();
    }

    public void add(int[] x, float val)
    {
        list.add(new MatEntry(x, val));
    }

    void decompose(ArrayList<float[][]> param)
    {
        ArrayList<float[][]> param2 = new ArrayList<>();
        for (float[][] x : param) {
            int n = x.length;
            param2.add(new float[rank][n]);
        }

        double eps = 1e-6;
        for (int i = 0; i < rank; ++i) {
            ArrayList<float[]> a = new ArrayList<>();
            for (int k = 0; k < dim; ++k) {
                a.add(Utils.getRandomUnitVector(N[k]));
            }

            int iter;
            double norm = 0.0;
            double lastnorm = Double.POSITIVE_INFINITY;

            for (iter = 0; iter < MAX_ITER; ++iter) {
                for (int k = 0; k < dim; ++k) {
                    float[] b = a.get(k);
                    for (int j = 0; j < N[k]; ++j)
                        b[j] = 0;
                    for (MatEntry e : list) {
                        double s = e.val;
                        for (int l = 0; l < dim; ++l)
                            if (l != k)
                                s *= a.get(l)[e.x[l]];
                        b[e.x[k]] += s;
                    }
                    for (int j = 0; j < i; ++j) {
                        double dot = 1;
                        for (int l = 0; l < dim; ++l)
                            if (l != k)
                                dot *= Utils.dot(a.get(l), param2.get(l)[j]);
                        for (int p = 0; p < N[k]; ++p)
                            b[p] -= dot*param2.get(k)[j][p];
                    }

                    if (k < dim-1){
                        Utils.normalize(b);
                    } else {
                        norm = Math.sqrt(Utils.squaredSum(b));
                    }
                }
                if (lastnorm != Double.POSITIVE_INFINITY && Math.abs(norm-lastnorm) < eps)
                    break;
                lastnorm = norm;
            }
            if (iter >= MAX_ITER) {
                System.out.printf("\tWARNING: Power method didn't converge." +
                        "rankFirstOrderTensor=%d sigma=%f%n", i, norm);
            }
            if (Math.abs(norm) <= eps) {
                System.out.printf("\tWARNING: Power method has nearly-zero sigma. rankFirstOrderTensor=%d%n",i);
            }
            System.out.printf("\t%.2f", norm);
            for (int k = 0; k < dim; ++k)
                param2.get(k)[i] = a.get(k);
        }

        for (int i = 0; i < param.size(); ++i) {
            float[][] x = param.get(i);
            float[][] y = param2.get(i);
            int n = x.length;
            for (int u = 0; u < n; ++u)
                for (int v = 0; v < rank; ++v)
                    x[u][v] = y[v][u];
        }
        System.out.println();
    }

}

class MatEntry
{
    int[] x;
    double val;

    MatEntry(int[] x, double val)
    {
        this.x = x.clone();
        this.val = val;
    }
}