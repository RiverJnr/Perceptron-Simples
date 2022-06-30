package model;

import java.util.Random;

public class Perceptron {

    private double ni;
    private double[][] pesos;
    // private double[][] pesos = {{0},{0.3},{0.1}};

    public Perceptron(int qtdAmostra, int arrayTeta, Double ni) {
        this.ni = ni;
        this.pesos = new double[qtdAmostra + 1][arrayTeta];
        Random rand = new Random();
        for (int i = 0; i < pesos.length; i++) {
            for (int j = 0; j < pesos[0].length; j++) {
                double w = rand.nextDouble(0.03 * 2) - 0.03;
                this.pesos[i][j] = w;
            }
        }
    }

    public Double[] learn(Double[] Xamostras, Double[] Yamostras) {
        double[] x = new double[Xamostras.length + 1];
        x[x.length - 1] = 1;
        for (int i = 0; i < x.length - 1; i++) {
            x[i] = Xamostras[i];
        }
        Double[] teta = new Double[Yamostras.length];

        // Percorre cada coluna da saida Y
        for (int j = 0; j < teta.length; j++) {
            double u = 0;
            // Percorre cada linha da saida Y
            for (int i = 0; i < x.length; i++) {
                u += x[i] * pesos[i][j];
            }
            teta[j] = this.sigmoidal(u);
        }

        for (int j = 0; j < pesos[0].length; j++) {
            for (int i = 0; i < pesos.length; i++) {
                pesos[i][j] += ni * (Yamostras[j] - teta[j]) * x[i];
            }
        }
        return teta;
    }

    public double[][] getPesos() {
        return this.pesos;
    }

    private double sigmoidal(double u) {
        return 1 / (1 + Math.exp(-u));
    }
}