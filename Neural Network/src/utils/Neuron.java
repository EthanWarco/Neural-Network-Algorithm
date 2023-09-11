package utils;

import java.util.HashMap;
import java.util.Map;

public class Neuron {
	
	//the connections are only the ones that flow forwards, no connections with the layer behind it are known
	//the velocities map is for previous weight increments, used for momentum
	public final Map<Neuron, Double> connections = new HashMap<Neuron, Double>();
	public final Map<Neuron, Double> velocities = new HashMap<Neuron, Double>();
	public final Map<Neuron, Double> gradients = new HashMap<Neuron, Double>();
	public double bias;
	public double biasGradient;
	public double dcost_dout;
	public double biasVelocity;
	public double value;
	
	public Neuron() {
		this.bias = 0;
		this.biasVelocity = 0;
	}
	
}
