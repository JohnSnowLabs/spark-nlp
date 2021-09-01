/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

import java.util.Random;

public final class Utils {

    private Utils() { throw new IllegalStateException("Utility class"); }

    private static Random rnd = new Random(System.currentTimeMillis());

    public static int log2(long x)
    {
        long y = 1;
        int i = 0;
        while (y < x) {
            y = y << 1;
            ++i;
        }
        return i;
    }

    public static float[] getRandomUnitVector(int length)
    {
        float[] vec = new float[length];
        float sum = 0;
        for (int i = 0; i < length; ++i) {
            vec[i] = (float) (rnd.nextFloat() - 0.5);
            sum += vec[i] * vec[i];
        }
        float invSqrt = (float) (1.0 / Math.sqrt(sum));
        for (int i = 0; i < length; ++i)
            vec[i] *= invSqrt;
        return vec;
    }

    public static float squaredSum(float[] vec)
    {
        float sum = 0;
        for (float aVec : vec) sum += aVec * aVec;
        return sum;
    }

    public static void normalize(float[] vec)
    {
        double coeff = 1.0 / Math.sqrt(squaredSum(vec));
        for (int i = 0, N = vec.length; i < N; ++i)
            vec[i] *= coeff;
    }

    public static float max(float[] vec)
    {
        float max = Float.NEGATIVE_INFINITY;
        for (float aVec : vec) max = Math.max(max, aVec);
        return max;
    }

    public static float min(float[] vec)
    {
        float min = Float.POSITIVE_INFINITY;
        for (float aVec : vec) min = Math.min(min, aVec);
        return min;
    }

    public static float dot(float[] u, float[] v)
    {
        float dot = 0.0f;
        for (int i = 0, N = u.length; i < N; ++i)
            dot += u[i]*v[i];
        return dot;
    }

}
