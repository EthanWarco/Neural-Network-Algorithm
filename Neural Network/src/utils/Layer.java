package utils;

import java.util.Arrays;
import java.util.Random;

public class Layer {
	
	public final Neuron[] neurons;
	public final boolean biases;
	public final String activation;
	public Layer outputLayer;
	
	public static final String SOFTMAX = "softmax";
	public static final String SIGMOID = "sigmoid";
	public static final String RELU = "relu";
	public static final String TANH = "tanh";
	
	public Layer(int neurons, String activation, boolean biases) {
		this.biases = biases;
		this.activation = activation;
		this.neurons = new Neuron[neurons];
		for(int i = 0; i < neurons; i++) {
			this.neurons[i] = new Neuron();
		}
	}
	
	
	public void activate() {
		if(activation == SOFTMAX) {
			double[] exp = new double[neurons.length];
			for(int i = 0; i < neurons.length; i++) exp[i] = Math.exp(neurons[i].value);
			double expsum = Arrays.stream(exp).sum();
			for(int i = 0; i < exp.length; i++) neurons[i].value = exp[i]/expsum;
		} else if(activation == SIGMOID) {
			for(int i = 0; i < neurons.length; i++) neurons[i].value = 1.0/(1.0 + Math.exp(-neurons[i].value));
		} else if(activation == RELU) {
			for(int i = 0; i < neurons.length; i++) neurons[i].value = Math.max(0.0, neurons[i].value);
		} else if(activation == TANH) {
			for(int i = 0; i < neurons.length; i++) neurons[i].value = 2.0/(1.0 + Math.exp(-2*neurons[i].value)) - 1.0;
		}
	}
	
	public void backPropagate(double[] actualValues) {
		for(int i = 0; i < outputLayer.neurons.length; i++) {
			Neuron output = outputLayer.neurons[i];
			
			
			double der = 0.0;
			if(outputLayer.activation == SOFTMAX) der = output.value - actualValues[i];
			else if(outputLayer.activation == SIGMOID) der = (output.value*(1.0-output.value))*output.dcost_dout;
			else if(outputLayer.activation == RELU && output.value >= 0) der = output.dcost_dout;
			else if(outputLayer.activation == TANH) der = (1.0 - output.value*output.value)*output.dcost_dout;
			
			for(int j = 0; j < neurons.length; j++) {
				Neuron input = neurons[j];
				double weight = input.connections.get(output);
				
				input.dcost_dout += der*weight;
				input.gradients.replace(output, der*input.value);
			}
			
			
			output.biasGradient = der;
		}
	}
	
	public void updateParams(Optimizer optimizer, int batchSize) {
		if(optimizer instanceof SGD) {
			SGD gd = (SGD) optimizer;
			for(int i = 0; i < outputLayer.neurons.length; i++) {
				Neuron output = outputLayer.neurons[i];
				
				for(int j = 0; j < neurons.length; j++) {
					Neuron input = neurons[j];
					double weight = input.connections.get(output);
					double weightMomentum = input.velocities.get(output)*gd.momentum;
					
					input.connections.replace(output, weight - gd.lr*(input.gradients.get(output)/batchSize) + weightMomentum);
					input.velocities.replace(output, weight - input.connections.get(output));
				}
				
				double biasMomentum = output.biasVelocity*gd.momentum;
				if(outputLayer.biases) {
					double bias = output.bias;
					output.bias = bias - gd.lr*(output.biasGradient/batchSize) + biasMomentum;
					output.biasVelocity = bias - output.bias;
				}
			}
		}
	}
	
	
	//uses xavier initialization for weights
	public void fullyConnect(Layer layer) {
		outputLayer = layer;
		Random rand = new Random();
		double std = Math.sqrt(neurons.length);
		for(int i = 0; i < neurons.length; i++) {
			for(int j = 0; j < layer.neurons.length; j++) {
				neurons[i].connections.put(layer.neurons[j], rand.nextGaussian()/std);
				neurons[i].velocities.put(layer.neurons[j], 0.0);
				neurons[i].gradients.put(layer.neurons[j], 0.0);
			}
		}
	}
	
}