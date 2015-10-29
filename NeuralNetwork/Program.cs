using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    class Program {
        // Size of input vector
        static readonly int dIn = 2;
        // Size of hidden layer
        static readonly int d1 = 3;
        // Size of output vector
        static readonly int dOut = 1;
        // Weights matrix between input and hidden layer
        static double[] W1 = new double[dIn * d1];
        // Weights change matrix for batch update mode
        static double[] W1ChangeCache = initW1ChangeCache();
        // Bias between input and hidden layer
        static double[] b1 = new double[d1];
        // Bias change vector for batch update mode
        static double[] b1ChangeCache = initB1ChangeCache();
        // Weights matrix between hidden layer and output
        static double[] W2 = initW2ChangeCache();
        // Weights change matrix for batch update mode
        static double[] W2ChangeCache = new double[d1 * dOut];
        // Bias between hidden layer and output
        static double[] b2 = new double[dOut];
        // Bias change vector for batch update mode
        static double[] b2ChangeCache = initB2ChangeCache();
        // Output values of the hidden layer neurons
        static double[] hiddenLayerOutput;

        // How many times should we train the whole training set?
        //static readonly int epochs = 10000;
        static readonly double errorTarget = 0.01;
        // Learning rate / alpha
        static readonly double learningRate = 1;

        static void Main(string[] args) {
            trainXor();
        }

        private static void trainXor() {
            // x1, x2, y
            int[] trainingData = new int[] {
                0, 0, 0,
                0, 1, 1,
                1, 0, 1,
                1, 1, 0,
            };
            int trainingDataSetLength = 3;

            // Init weights
            initWeights(W1, dIn, d1);
            initWeights(W2, d1, dOut);
            b1[0] = 1.0;
            b1[1] = 1.0;
            b1[2] = 1.0;
            b2[0] = 1.0;
            //W1[0] = 0.3;
            //W1[1] = 0.2;
            //W1[2] = 0.8;
            //W1[3] = 0.9;
            //W1[4] = 0.5;
            //W1[5] = 0.1;
            //W2[0] = 0.2;
            //W2[1] = 0.3;
            //W2[2] = 0.8;

            int trainingDataSets = trainingData.Length / trainingDataSetLength;
            double averageError = 1.0;
            int epoch = 0;
            //for (int epoch = 1; epoch <= epochs; ++epoch) {
            while (averageError > errorTarget) {
                ++epoch;
                
                // Print epoch and weights information.
                //Console.WriteLine("*** Epoch " + epoch + " ***");
                //Console.WriteLine("W1:");
                //print(W1, d1);
                //Console.WriteLine("W2:");
                //print(W2, dOut);

                // Calculate and print average error.
                double errorSum = 0;
                for (int trainingDataSet = 0; trainingDataSet < trainingDataSets; ++trainingDataSet) {
                    int startIndex = trainingDataSet * trainingDataSetLength;
                    double[] x = new double[] { trainingData[startIndex], trainingData[startIndex + 1] };
                    double y = f(x);
                    double t = trainingData[startIndex + 2];
                    double E = y - t;
                    errorSum += (E >= 0 ? E : -E);
                }
                averageError = errorSum / trainingDataSets;
                //Console.WriteLine("Average error = " + averageError);

                for (int trainingDataSet = 0; trainingDataSet < trainingDataSets; ++trainingDataSet) {
                    int startIndex = trainingDataSet * trainingDataSetLength;
                    double[] x = new double[] {trainingData[startIndex], trainingData[startIndex + 1]};
                    // Forward pass: calculate output with current weights
                    double y = f(x);
                    // Target output
                    double t = trainingData[startIndex + 2];

                    // Calculate stochastic gradient descent.
                    // Online weights update.
                    calcAndUpdateWeights(x, y, t, false);

                    // Batch update, part 1
                    // Didn't get XOR approximation to work with batch update.
                    //calcAndUpdateWeights(x, y, t, true);

                    //Console.WriteLine("x:");
                    //print(x, dIn);
                    //Console.WriteLine("y = f(x) = " + y);
                    //Console.WriteLine("t = " + t);
                }

                // Batch update, part 2
                //updateWeights();
            }

            Console.WriteLine("Result after " + epoch + " epochs:");
            for (int trainingDataSet = 0; trainingDataSet < trainingDataSets; ++trainingDataSet) {
                int startIndex = trainingDataSet * trainingDataSetLength;
                double[] x = new double[] { trainingData[startIndex], trainingData[startIndex + 1] };
                double y = f(x);
                double t = trainingData[startIndex + 2];
                Console.WriteLine("x:");
                print(x, dIn);
                Console.WriteLine("y = f(x) = " + y);
                Console.WriteLine("t = " + t);
            }
        }

        // Forward pass (calculate output)
        // g(g(x * W1) * W2)
        //
        // TODO: Apply g() to output as well? They do it in the tutorial
        // at https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf.
        // Changes the results: Big jump in the first step, but takes longer
        // to be really precise, e.g. p20 example, get to 0.5000xxxxxxx:
        // Without g(): epoch 34
        // With g(): epoch 137
        // 
        // XOR results: Look a lot faster with g() applied.
        // Applying g() seems to make more than sense than not applying it.
        // But that's just me.
        private static double f(double[] x) {
            hiddenLayerOutput = g(add(multiply(x, W1, 1, dIn, d1), b1, 1, d1));
            return g(add(multiply(hiddenLayerOutput, W2, 1, d1, dOut), b2, 1, dOut))[0];
        }

        // ReLU
        // x = vector
        //private static double[] g(double[] x) {
        //    double[] y = new double[x.Length];
        //    for (int i = 0; i < x.Length; ++i) {
        //        y[i] = x[i] < 0 ? 0 : x[i];
        //    }
        //    return y;
        //}

        // Activation function
        // Logistic function / sigmoid
        // x = vector
        private static double[] g(double[] x) {
            double[] y = new double[x.Length];
            for (int i = 0; i < x.Length; ++i) {
                y[i] = 1 / (1 + Math.Pow(Math.E, -x[i]));
            }
            return y;
        }

        // Hinge (binary)
        //private static double L(double y, double t) {
        //    double loss = 1 - y * t;
        //    return loss < 0 ? 0 : loss;
        //}

        private static void initWeights(double[] W, int rows, int columns) {
            xavierInit(W, rows, columns);
        }
        // Xavier initialization
        private static void xavierInit(double[] W, int rows, int columns) {
            Random r = new Random();
            double max = Math.Sqrt(6) / Math.Sqrt(rows + columns);
            double min = -max;
            double range = max - min;
            for (int i = 0; i < W.Length; ++i) {
                W[i] = r.NextDouble() * range + min;
            }
        }

        // Error function:
        // E = t – y
        // Using the derivative of the logistic function:
        // g(x) * (1 - g(x))
        // deltaOutput, deltaHidden, updateWeights based on:
        // https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf p. 19
        private static double deltaOutput(double y, double t) {
            return y * (1 - y) * (t - y);
        }
        // Error function for hidden layer
        private static double deltaHidden(double dy, int i) {
            return hiddenLayerOutput[i] * (1 - hiddenLayerOutput[i]) * dy * W2[i];
        }
        // Calculate stochastic gradient descent.
        // caching = false: update weights online
        // caching = true: save weights to cache
        private static void calcAndUpdateWeights(double[] x, double y, double t, bool caching) {
            double[] myW1 = caching ? W1ChangeCache : W1;
            double[] myW2 = caching ? W2ChangeCache : W2;
            double[] myB1 = caching ? b1ChangeCache : b1;
            double[] myB2 = caching ? b2ChangeCache : b2;

            // Output layer error gradient
            double dy = deltaOutput(y, t);
            // W2 weights update
            for (int i = 0; i < myW2.Length; ++i) {
                myW2[i] += learningRate * dy * hiddenLayerOutput[i];
            }
            // b2 update
            for (int i = 0; i < myB2.Length; ++i) {
                myB2[i] += learningRate * dy;
            }

            // Hidden layer error gradient
            double[] dh = new double[d1];
            for (int i = 0; i < d1; ++i) {
                dh[i] = deltaHidden(dy, i);
            }
            // W1 weights update
            for (int i = 0; i < dIn; ++i) {
                for (int j = 0; j < d1; ++j) {
                    // W1[i][j]
                    myW1[i * d1 + j] += learningRate * dh[j] * x[i];
                }
            }
            // b1 update
            for (int i = 0; i < myB1.Length; ++i) {
                myB1[i] += learningRate * dh[i];
            }
        }
        // Apply cached weight changes, reset cache.
        private static void updateWeights() {
            for (int i = 0; i < W1.Length; ++i) {
                W1[i] += W1ChangeCache[i];
            }
            for (int i = 0; i < W2.Length; ++i) {
                W2[i] += W2ChangeCache[i];
            }
            initW1ChangeCache();
            initW2ChangeCache();
        }
        // Reset W1 cache.
        private static double[] initW1ChangeCache() {
            return new double[dIn * d1];
        }
        // Reset W2 cache.
        private static double[] initW2ChangeCache() {
            return new double[d1 * dOut];
        }
        // Reset b1 cache.
        private static double[] initB1ChangeCache() {
            return new double[d1];
        }
        // Reset b2 cache.
        private static double[] initB2ChangeCache() {
            return new double[dOut];
        }

        // TODO in-situ would be faster
        private static double[] add(double[] A, double[] B, int rows, int columns) {
            double[] C = new double[rows * columns];
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < columns; ++j) {
                    int index = i * columns + j;
                    C[index] = A[index] + B[index];
                }
            }
            return C;
        }

        // A = n x m row-major matrix / vector
        // B = m x p row-major matrix / vector
        // AB = n x p row-major matrix / vector
        private static double[] multiply(double[] A, double[] B, int n, int m, int p) {
            double[] AB = new double[n * p];
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < p; ++j) {
                    double sum = 0;
                    for (int k = 0; k < m; ++k) {
                        // A[i][k] * B[k][j]
                        sum += A[i * m + k] * B[k * p + j];
                    }
                    // AB[i][j]
                    AB[i * p + j] = sum;
                }
            }
            return AB;
        }

        private static void print(double[] M, int columns) {
            for (int i = 0; i < M.Length; ++i) {
                Console.Write(M[i] + ", ");
                if (i % columns == (columns - 1)) {
                    Console.WriteLine();
                }
            }
        }

        private static void addTest() {
            double[] A = new double[] { 1, 2, 3, 7, 8, 9 };
            double[] B = new double[] { 5, 6, 7, 3, 4, 5 };
            // Expected result:
            //  6  8 10
            // 10 12 14
            double[] C = add(A, B, 2, 3);
            print(C, 3);
        }
        private static void multiplyTest() {
            double[] A = new double[] { 1, 4, 6 };
            double[] B = new double[] { 2, 3, 5, 8, 7, 9 };
            // Expected result: (64, 89)
            double[] AB = multiply(A, B, 1, 3, 2);
            foreach (double d in AB) {
                Console.Write(d + ", ");
            }
        }
    }
}
