import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

import kohonen.LearningData;
import kohonen.LearningDataModel;
import kohonen.WTMLearningFunction;
import learningFactorFunctional.ConstantFunctionalFactor;
import learningFactorFunctional.GaussFunctionalFactor;
import learningFactorFunctional.LearningFactorFunctionalModel;
import metrics.EuclidesMetric;
import metrics.MetricModel;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import network.DefaultNetwork;
import network.KohonenNeuron;
import network.NetworkModel;
import network.NeuronModel;
import topology.GaussNeighbourhoodFunction;
import topology.MatrixTopology;
import topology.NeighbourhoodFunctionModel;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;


public class SOM {
	
	public void printToHTML(int[] array, int[] cluster, int col, int row, String filename){
		String html = "<html> <head> SOM </head> <body>";
		
		html += "<table>";
		
		for(int i=0;i<array.length; i = i + col){
			
			html += "<tr>";
			
			
			for(int j=i; j<i+col; j++){
	
				if(array[j] == 0){
					if(cluster[j] == 0){
						html += "<td> <span style='color: green'>" +array[j] + " </span></td> ";
					} else {
						html += " <td><span style='color: orange'>" +array[j] + "</span></td> ";
					}
				} else {
					if(cluster[j] == 0){
						html += " <td><span style='color: green'>" +array[j] + "</span></td> ";
					} else {
						html += " <td><span style='color: orange'>" +array[j] + " </span></td> ";
					}
				}
			}
			
		
			html += "</tr>";
			
		}
		
		html += "</table></body></html>";
		File file = new File("./"+filename+".html");
	
			try {
				file.createNewFile();
				BufferedWriter bw = new BufferedWriter(new FileWriter(file));
				bw.write(html);
				bw.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		
	}
	
	public int bestLabel(String filename,NeuronModel network, MetricModel metric) throws FileNotFoundException{
		
		File file = new File(filename);
		Scanner s = new Scanner(file);
		String[] temp = null;
		double min = 10000;
		String current = null;
		int index = -1;
		while(s.hasNextLine()){
			String list = s.nextLine();
			temp = list.split(",");
			double[] vector = new double[network.getWeight().length];
			for(int i=0; i<vector.length;i++){
				vector[i] = Double.valueOf(temp[i]).doubleValue();
			}
			
			
			double sum = metric.getDistance(vector, network.getWeight());
			
			if(sum<min){
				min = sum;
				current = temp[temp.length-1];
			}
		}
		System.out.println(current);
		s.close();
		return Integer.valueOf(current).intValue();
		
	}
	
	public FastVector getHeader(String filename) throws FileNotFoundException{
		File file = new File(filename);
		Scanner s = new Scanner(file);
		String[] temp = s.nextLine().split(",");
		FastVector header = new FastVector();
		for(int i=0;i<temp.length; i++){
			Attribute attribute = new Attribute(temp[0]);
			header.addElement(attribute);
		}
		
		return header;
	}
	public void runSOM(){
		double[] maxWeight = {2, 1,	2,	1,	2,	2,	2,	1,	2,	1,	2,	2,	2,	1,	1,	1,	2,	2,	2,	1,	2,	1,	2,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	2,	1,	1,	1,	1,	1,	2,	1,	1,	1,	2,	1,	1,	3,	2,	1,	2,	1,	0.926353555,	-0.006086259,	-0.424889965,	1};
		double[] maxWeight2 = {41,	10,	5,	6,	10,	9,	9,	5,	9,	9,	7,	9,	9,	9,	9,	9,	9,	9,	9,	5,	9,	9,	9,	9,	9,	9,	9,	9,	9,	9,	9,	9,	7,	9,	9,	9,	9,	9,	9,	9,	9, 9,	8,	3,	6,	4,	8,	7,	7,	9,	5,	6,	6,	6,	9,	6,	3,	7,	8, 3, 6,	1,	 6,	5,	2,	5, 1,	7,	4,	8,	3,	3,	4,	6,	2,	8,	1,	1,	2,	7,	1,	2,	3,	2,	2,	1};

		String filename = "./dataset/spambased-normalized-noheader.csv";
		String filenameHeader = "./dataset/spambased_normalized_finally.csv";
		String filename2 = "./dataset/spambased_normalized_finally-withlabel.csv";
		
		String filename3 = "./dataset/tic_withLabel_noMissingValues3-nolabel.csv";
		String filenameHeader2 = "./dataset/tic_withLabel_noMissingValues.csv";
		String filename4 = "./dataset/tic_withLabel_noMissingValues3.csv";
		
		LearningDataModel model = new LearningData(filename3);
		MatrixTopology matrix = new MatrixTopology(10,10);
		NetworkModel network = new DefaultNetwork(maxWeight2.length,maxWeight2,matrix);
		MetricModel euclides = new EuclidesMetric();
		LearningFactorFunctionalModel learningModel = new GaussFunctionalFactor();
		NeighbourhoodFunctionModel neighborModel = new GaussNeighbourhoodFunction(1);

		WTMLearningFunction wtm = new WTMLearningFunction(network,20,euclides,model,learningModel,neighborModel);
		wtm.learn();
		System.out.println(network);
		int[] labelled = new int[network.getNumbersOfNeurons()];
		for(int i=0;i<network.getNumbersOfNeurons();i++){
			try {
			//	System.out.println(bestLabel(filename2,network.getNeuron(i),euclides));
				//labelled[i] = bestLabel(filename2,network.getNeuron(i), euclides);
				labelled[i] = bestLabel(filename4,network.getNeuron(i), euclides);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		for(int i=1; i<= network.getNumbersOfNeurons();i++){
			
				
			System.out.print(labelled[i-1] + " ");
			if(i%network.getTopology().getColNumber() == 0 && i !=1)
			System.out.print("\n");
		}
		
		
		int[] cluster = new int[network.getNumbersOfNeurons()];
		Instances instances = null;
		try {
			//instances = new Instances("SOM", getHeader(filenameHeader),network.getNumbersOfNeurons());
			instances = new Instances("SOM", getHeader(filenameHeader2),network.getNumbersOfNeurons());
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	
		for(int i=0 ; i<network.getNumbersOfNeurons(); i++){
			Instance instance = new Instance(1,network.getNeuron(i).getWeight());
			instances.add(instance);
			
		}
		
		SimpleKMeans kmean = new SimpleKMeans();
		
		try {
			kmean.setSeed(3);
			kmean.setNumClusters(2);
			kmean.setPreserveInstancesOrder(true);
			kmean.buildClusterer(instances);
			for(int i=0; i<network.getNumbersOfNeurons(); i++){
				cluster[i] = kmean.clusterInstance(instances.instance(i));
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	
		printToHTML(labelled,cluster, network.getTopology().getRowNumber(), network.getTopology().getColNumber(),"SOM-Cluster");
	
	}
}
