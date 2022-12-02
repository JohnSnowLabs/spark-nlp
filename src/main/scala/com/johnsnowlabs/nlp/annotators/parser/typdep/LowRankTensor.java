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

import com.johnsnowlabs.nlp.annotators.parser.typdep.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

public class LowRankTensor {

    private Logger logger = LoggerFactory.getLogger("TypedDependencyParser");
    private int dim;
    private int rank;
    private int[] N;
    private ArrayList<MatEntry> list;

    private static final int MAX_ITER = 1000;

    LowRankTensor(int[] N, int rank) {
        this.N = N.clone();
        dim = N.length;
        this.rank = rank;
        list = new ArrayList<>();
    }

    public void add(int[] x, float val) {
        list.add(new MatEntry(x, val));
    }

    void decompose(ArrayList<float[][]> param) {
        ArrayList<float[][]> param2 = new ArrayList<>();
        for (float[][] x : param) {
            int n = x.length;
            param2.add(new float[rank][n]);
        }

        double eps = 1e-6;
        for (int i = 0; i < rank; ++i) {
            ArrayList<float[]> aArrayVariable = new ArrayList<>();
            for (int k = 0; k < dim; ++k) {
                aArrayVariable.add(Utils.getRandomUnitVector(N[k]));
            }

            int iter;
            double norm = 0.0;
            double lastnorm = Double.POSITIVE_INFINITY;

            for (iter = 0; iter < MAX_ITER; ++iter) {
                for (int k = 0; k < dim; ++k) {
                    float[] b = aArrayVariable.get(k);
                    for (int j = 0; j < N[k]; ++j)
                        b[j] = 0;
                    for (MatEntry matentry : list) {
                        double s = matentry.val;
                        for (int l = 0; l < dim; ++l)
                            if (l != k) {
                                s *= aArrayVariable.get(l)[matentry.x[l]];
                            }
                        b[matentry.x[k]] += s;
                    }
                    for (int j = 0; j < i; ++j) {
                        double dot = 1;
                        for (int l = 0; l < dim; ++l)
                            if (l != k)
                                dot *= Utils.dot(aArrayVariable.get(l), param2.get(l)[j]);
                        for (int p = 0; p < N[k]; ++p)
                            b[p] -= dot * param2.get(k)[j][p];
                    }

                    if (k < dim - 1) {
                        Utils.normalize(b);
                    } else {
                        norm = Math.sqrt(Utils.squaredSum(b));
                    }
                }
                if (lastnorm != Double.POSITIVE_INFINITY && Math.abs(norm - lastnorm) < eps)
                    break;
                lastnorm = norm;
            }
            if (iter >= MAX_ITER) {
                logger.warn("Power method didn't converge." +
                        "rankFirstOrderTensor=%d sigma=%f%n", i, norm);
            }
            if (Math.abs(norm) <= eps && logger.isDebugEnabled()) {
                logger.warn(String.format("Power method has nearly-zero sigma. rankFirstOrderTensor=%d%n", i));
            }
            if (logger.isDebugEnabled()) {
                logger.debug(String.format("norm: %.2f", norm));
            }
            for (int k = 0; k < dim; ++k)
                param2.get(k)[i] = aArrayVariable.get(k);
        }

        for (int i = 0; i < param.size(); ++i) {
            float[][] x = param.get(i);
            float[][] y = param2.get(i);
            int n = x.length;
            for (int u = 0; u < n; ++u)
                for (int v = 0; v < rank; ++v)
                    x[u][v] = y[v][u];
        }
    }

}

class MatEntry {
    int[] x;
    double val;

    MatEntry(int[] x, double val) {
        this.x = x.clone();
        this.val = val;
    }
}