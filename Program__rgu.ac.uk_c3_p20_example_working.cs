using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    class Program {
        static readonly int dIn = 2;
        static readonly int d1 = 2;
        static readonly int dOut = 1;
        static double[] W1 = new double[dIn * d1];
        static double[] W2 = new double[d1 * dOut];
        static double[] hiddenLayerOutput;

        static readonly int epochs = 1000;
        static readonly double learningRate = 1;

        static void Main(string[] args) {
            trainXor();
        }

        private static void trainXor() {
            double[] trainingData = new double[] {
                0.35, 0.9, 0.5,
            };
            int trainingDataSetLength = 3;

            // Init weights
            W1[0] = 0.1;
            W1[1] = 0.4;
            W1[2] = 0.8;
            W1[3] = 0.6;
            W2[0] = 0.3;
            W2[1] = 0.9;

            for (int epoch = 1; epoch <= epochs; ++epoch) {
                Console.WriteLine("*** Epoch " + epoch + " ***");
                Console.WriteLine("W1:");
                print(W1, d1);
                Console.WriteLine("W2:");
                print(W2, dOut);
                for (int trainingDataSet = 0; trainingDataSet < trainingData.Length / trainingDataSetLength; ++trainingDataSet) {
                    int startIndex = trainingDataSet * trainingDataSetLength;
                    double[] x = new double[] {trainingData[startIndex], trainingData[startIndex + 1]};
                    double y = f(x);
                    double t = trainingData[startIndex + 2];
                    updateWeights(x, y, t);

                    Console.WriteLine("x:");
                    print(x, dIn);
                    Console.WriteLine("y = f(x) = " + y);
                    Console.WriteLine("t = " + t);
                }
            }
        }

        // g(x * W1) * W2
        // TODO: Apply g() to output as well? They do it in the tutorial
        // at https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf.
        // Changes the results: Big jump in the first step, but takes longer
        // to be really precise, e.g. 0.5000xxxxxxx:
        // Without g(): epoch 34
        // With g(): epoch 137
        //
        // TODO: Does it / can it make sense to use non-binary neuron
        // outputs like me // what is this threshold thing all about?
        private static double f(double[] x) {
            hiddenLayerOutput = g(multiply(x, W1, 1, dIn, d1));
            return multiply(hiddenLayerOutput, W2, 1, d1, dOut)[0];
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

        // Error function:
        // E = t – y
        // Using the derivative of the logistic function:
        // g(x) * (1 - g(x))
        // deltaOutput, deltaHidden, updateWeights based on:
        // https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf p. 19
        private static double deltaOutput(double y, double t) {
            return y * (1 - y) * (t - y);
        }

        // TODO pass deltaOutput
        private static double deltaHidden(double y, double t, int i) {
            return hiddenLayerOutput[i] * (1 - hiddenLayerOutput[i]) * deltaOutput(y, t) * W2[i];
        }

        private static void updateWeights(double[] x, double y, double t) {
            // Output layer error gradient
            double dy = deltaOutput(y, t);
            // W2 weights update
            for (int i = 0; i < W2.Length; ++i) {
                W2[i] += learningRate * dy * hiddenLayerOutput[i];
            }

            // Hidden layer error gradient
            double[] dh = new double[d1];
            for (int i = 0; i < d1; ++i) {
                dh[i] = deltaHidden(y, t, i);
            }
            // W1 weights update
            for (int i = 0; i < dIn; ++i) {
                for (int j = 0; j < d1; ++j) {
                    // W1[i][j]
                    W1[i * d1 + j] += learningRate * dh[j] * x[i];
                }
            }
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
