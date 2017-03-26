package naive_bayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * @author Kundan Kumar
 *
 */
public class NaiveBayes {


	private String SPACE=" ";
	private String COMMA=",";
	private int mTargetIdx=-1;
	private boolean KEEP_DUPLICATES = true;//false;

	/** The unique attributes. */
	private List<String> mAttributes = 
			new ArrayList<String>();

	/** Stores the target class priors */
	private HashMap<String, Double> mClassPriors = null; 
			
	/** The attributes to unique values map. */
	private HashMap<String, List<String>> mAtbToUnqVals= 
			new HashMap<String,List<String>>();

	/** The training data read from file and pre-processed.*/
	private List<HashMap<String,String> > mTrainingData =
			new ArrayList<HashMap<String, String>>();

	/** The training data read from file and pre-processed.*/
	private List<HashMap<String,String> > mTestData =
			new ArrayList<HashMap<String, String>>();
	
	/** Holds priors for the attribute values.*/
	private HashMap<String,HashMap<String,HashMap<String, Double>>>
					mTgtAtbvToAtbtsToAtbtvToPrior =null;
	
	/** Target class occurrence count */
	private HashMap<String,Integer> mTargetAtbValToCount = 
			new HashMap<String,Integer> ();

	/** Stores he final classification. */
	private List<String> mPredictions = null;

	/**
	 * Trains the classifier.
	 *
	 * @param trainDataPath the train data path
	 * @param delimiter the delimiter
	 * @param targetIdx the target idx
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	public void train(String trainDataPath, String delimiter, int targetIdx) 
			throws IOException {
		if(!(delimiter.equals(SPACE) || delimiter.equals(COMMA) )){
			System.out.println("Error >> Invalid delimiter specified.");
			return;
		}
		mTargetIdx = targetIdx;

		/* Read training data from file.*/
		readTrainingData(trainDataPath,delimiter);

		/* Remove spurious and duplicate data*/
		preProcessTrainingData();

		/* Calculate priors for class and attributes.*/
		buildClassifier();

	}
	
	
	/**
	 * Classifies using the trained classifier.
	 *
	 * @param testDataPath the test data path
	 * @param delimiter the delimiter
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	public void classify(String testDataPath, String delimiter) 
			throws IOException{
		if(mClassPriors==null){
			System.out.println("Invalid API sequence >> train() method not "
					+ "called.");
			return;
		}
		readAndStoreTestData(testDataPath,delimiter);
		
		mPredictions = new ArrayList<String>();
		for(int idx=0;idx<mTestData.size();idx++){
			HashMap<String, Double> result = new HashMap<String, Double>();
			
			HashMap<String, String> testData =  mTestData.get(idx);
					
			for(String aClass: mClassPriors.keySet()){
				double prob = Math.log(mClassPriors.get(aClass));
				
				for(String atb:testData.keySet()){
					if(mAttributes.get(mTargetIdx).equals(atb)){continue;}
					double atbPrior = 0;
					HashMap<String, Double> atbValToPrior = 
						mTgtAtbvToAtbtsToAtbtvToPrior.get(aClass).get(atb);
					
					if( atbValToPrior.containsKey(testData.get(atb))){
						atbPrior = mTgtAtbvToAtbtsToAtbtvToPrior.get(aClass).
								get(atb).get(testData.get(atb));
					}
					if(atbPrior==0){
						/*Calculate m-estimated probability*/
						double nc=0;
						double m=1;//Laplace Smoothing
						double n= mTargetAtbValToCount.get(aClass);
						double p= (double)1/(double)mAtbToUnqVals.get(atb).size();
						atbPrior = (nc+m*p)/(n+m);
						//System.out.println("\n\natbPrior:"+atbPrior+" p:"+p+" atb:"+atb);
					}
					prob = prob+Math.log(atbPrior);
				} 
				//System.out.println(prob);
				result.put(aClass, prob);
			}
			//System.out.println("\n");
			mPredictions.add(getPredictedClass(result));
		}
	}
	
	/**
	 * Evaluate.
	 *
	 * @param resultFilepath the result filepath
	 */
	public void evaluate(String resultFilepath){
		String dataToWrite= "";
		int correctCount=0;
		
		/* Generate the header.*/
		for(int idx=0;idx<mAttributes.size();idx++){
			dataToWrite=dataToWrite+" "+mAttributes.get(idx);
		}
		dataToWrite = dataToWrite+" "+mAttributes.get(mTargetIdx)+"(Predicted)\n";
		
		for(int idx=0;idx<mTestData.size();idx++){
			for(int cnt=0;cnt<mAttributes.size();cnt++){
				String atb = mAttributes.get(cnt);
				dataToWrite= dataToWrite+" "+mTestData.get(idx).get(atb);
			}
			
			correctCount = mTestData.get(idx).get(mAttributes.get(mTargetIdx)).
					equals(mPredictions.get(idx)) ? correctCount+1:correctCount;
			
			dataToWrite = dataToWrite+" "+mPredictions.get(idx)+"\n";
		}
		double accuracy = (double)correctCount/(double)mTestData.size();
		dataToWrite = dataToWrite+"Accuracy:"+correctCount+"/"+mTestData.size()+" = "+accuracy+"\n";
		System.out.println("\n\n---Predictions--\n"+dataToWrite);
	}
	
	/**
	 * Read and store test data.
	 *
	 * @param testDataPath the test data path
	 * @param delimiter the delimiter
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	private void readAndStoreTestData(String testDataPath, String delimiter) 
			throws IOException{
		/* Read file lines*/
		BufferedReader br= new BufferedReader(new FileReader(testDataPath));
		int lineCount=0;
		for(String line; (line = br.readLine()) != null;){
			line = preProcessLine(line);

			/* Continue if empty line*/
			if(line.length() <1){continue;}

			String[] colTokens = line.trim().toLowerCase().split(delimiter);
			/* NOTE: It is assumed that the test file has header exactly as the 
			 * training file, and in the first line.*/
			if(lineCount>0){
				HashMap<String,String> atbToAtbVals =
						new HashMap<String,String>();
				for(int idx=0; idx<colTokens.length;idx++){
					
					atbToAtbVals.put(mAttributes.get(idx),colTokens[idx]);
					
					updateAttributebUniqueVals(mAttributes.get(idx),
							colTokens[idx]);
				}
				mTestData.add(atbToAtbVals);
			}
			lineCount = lineCount+1;
		}
		br.close();
	}
	
	/**
	 * Builds the classifier.
	 */
	private void buildClassifier(){
		
		/* Calculate class priors.*/
		calculateClassPriors();
		
		/* Calculate priors of each attribute.*/
		calculatePriorsForEachAttribute();

	}
	
	/**
	 * Gets the predicted class.
	 *
	 * @param result the result
	 * @return the predicted class
	 */
	private String getPredictedClass(HashMap<String, Double> result){
		double max = 0;
		String predictedClass="";
		
		/* Initialize max*/
		for(String aClass:result.keySet()){
			max=result.get(aClass);
			predictedClass=aClass;
		}
		
		/* Find the predicted class*/
		for(String aClass:result.keySet()){
			if(result.get(aClass)>max){
				max = result.get(aClass);
				predictedClass=aClass;
			}
		}
		return predictedClass;
	}
	
	/**
	 * Calculate class priors.
	 */
	private void calculateClassPriors(){
		String targetClass= mAttributes.get(mTargetIdx);
		for(int idx=0; idx<mAtbToUnqVals.get(targetClass).size();idx++){
			String targetAtbVal = mAtbToUnqVals.get(targetClass).get(idx);
			for(int cnt=0; cnt<mTrainingData.size();cnt++){
				if(mTrainingData.get(cnt).get(targetClass).
						equals(targetAtbVal)){
					Integer prevVal = 0;
					if(mTargetAtbValToCount.containsKey(targetAtbVal)){
						prevVal = mTargetAtbValToCount.get(targetAtbVal);
					}
					mTargetAtbValToCount.put(targetAtbVal,prevVal+1);
				}
			}
		}
		
		mClassPriors = new HashMap<String, Double>();
		for(String targetAtb:mTargetAtbValToCount.keySet() ){
			Double prior = (double) (mTargetAtbValToCount.get(targetAtb)/
							(double)mTrainingData.size());
			mClassPriors.put(targetAtb,prior);
		}
	}

	/**
	 * Calculate priors for each attribute.
	 */
	private void calculatePriorsForEachAttribute(){
		mTgtAtbvToAtbtsToAtbtvToPrior =  new HashMap<>(); 
		String targetClass= mAttributes.get(mTargetIdx);
		
		/* Get class based priors for each attribute-values*/
		for(int idx=0; idx<mAtbToUnqVals.get(targetClass).size();idx++){

			HashMap<String,HashMap<String, Double>> atbToatbValCount =
					new HashMap<String,HashMap<String, Double>>();

			String targetAtbVal = mAtbToUnqVals.get(targetClass).get(idx);
			for(String atb:mAtbToUnqVals.keySet()){
				if(atb.equals(targetClass)){continue;}
				
				HashMap<String, Double> atbValToCount = 
						new HashMap<String, Double>();

				List<String> atbVals = mAtbToUnqVals.get(atb);
				for(int cnt=0;cnt<atbVals.size();cnt++){
					atbValToCount.put(atbVals.get(cnt), (double) 0);
					for(int m=0; m<mTrainingData.size();m++){
						String curAtb = mTrainingData.get(m).get(atb);
						String tgtAtb = mTrainingData.get(m).get(targetClass);
					  if( curAtb.equals(atbVals.get(cnt)) && 
							  tgtAtb.equals(targetAtbVal)){
							Double prevVal = 
									atbValToCount.get(atbVals.get(cnt));
							atbValToCount.put(atbVals.get(cnt), prevVal+1);
						}
					}
					
					Double prior = atbValToCount.get(atbVals.get(cnt))/
							(double)mTargetAtbValToCount.get(targetAtbVal);
					
					atbValToCount.put(atbVals.get(cnt), prior);
					atbToatbValCount.put(atb, atbValToCount);
				}
			}
			mTgtAtbvToAtbtsToAtbtvToPrior.put(targetAtbVal, atbToatbValCount);
		}
	}
	
	/**
	 * Prints the training data.
	 *
	 * @param dataIdx the data idx
	 */
	private void printTrainingData(int dataIdx){
		String data="";
		HashMap<String, String> vals= mTrainingData.get(dataIdx);
		for(String val: vals.keySet()){
			data= data+" "+vals.get(val);
		}
		System.out.println(data);
	}

	/**
	 * Pre process training data.
	 */
	private void preProcessTrainingData() {
		List<Integer> idxToRemove = new ArrayList<Integer>(); 

		for(int idx=0; idx<mTrainingData.size();idx++){
			HashMap<String,String> oneAtbToVal = mTrainingData.get(idx);

			for(int idx1=idx+1; idx1<mTrainingData.size();idx1++){
				boolean areAllAtbValSame = true;
				HashMap<String,String> atbToVal =  mTrainingData.get(idx1);

				/* Check if all the attributes except the target, are same.*/
				for(int idx2=0;idx2<mAttributes.size();idx2++){
					if(idx2==mTargetIdx){continue;}
					String atb = mAttributes.get(idx2);
					if(!oneAtbToVal.get(atb).equals(atbToVal.get(atb))){
						areAllAtbValSame = false;
						break;
					}
				}

				/* If all attributes are same, add to drop list.*/
				if(areAllAtbValSame){
					/* If target value of oneAtbToVal and atbToVal are       */ 
					/* different, drop both. If same, drop if duplicates are */
					/* not allowed.    */
					if(!oneAtbToVal.get( mAttributes.get(mTargetIdx)).
							equals(atbToVal.get( mAttributes.get(mTargetIdx)))){
						if(!idxToRemove.contains(idx)){
							idxToRemove.add(idx);	
						}
						if(!idxToRemove.contains(idx1)){
							idxToRemove.add(idx1);	
						}
					}
					if(!KEEP_DUPLICATES){
						if(!idxToRemove.contains(idx1)){
							idxToRemove.add(idx1);	
						}
					}

				}
			}
		}

		System.out.println("Duplicate rows:");
		System.out.println(idxToRemove.toString());

		/* Delete identified data from the list.*/
		int shift=0;
		for(int idx=0;idx<idxToRemove.size();idx++){
			mTrainingData.remove(idx-shift);
			shift=shift+1;
		}

		System.out.println("\n\n-----FINAL TRAINING DATA--------");
		System.out.println("Total Rows:"+mTrainingData.size());
		for(int idx=0; idx<mTrainingData.size();idx++){
			printTrainingData(idx);
		}
	}



	/**
	 * Sets the headers and transactions.
	 *
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	private void readTrainingData(String trainDataPath, String delimiter) 
			throws IOException{
		/* Read file lines*/
		BufferedReader br= new BufferedReader(new FileReader(trainDataPath));
		int lineCount=0;
		for(String line; (line = br.readLine()) != null;){
			line = preProcessLine(line);

			/* Continue if empty line*/
			if(line.length() <1){continue;}

			String[] colTokens = line.trim().toLowerCase().split(delimiter);
			if(lineCount==0){
				/*Set headers*/
				setAttributes(colTokens);
			}else{
				/* Add all training data.*/
				addTrainingData(colTokens);
			}
			lineCount = lineCount+1;
		}
		br.close();
	}

	/**
	 * Pre-process a line.
	 * - Replace multiple empty spaces by single empty space.
	 * @param line the line
	 * @return the string
	 */
	private String preProcessLine(String line){
		/*Replace multiple empty spaces by single empty space.*/
		line = line.replaceAll("\\s+", " ");
		return line;
	}

	/**
	 * Sets the attributes.
	 *
	 * @param colTokens the new attributes
	 */
	private void setAttributes(String[] colTokens) {
		for(int idx=0; idx< colTokens.length;idx++){
			mAttributes.add(colTokens[idx]);
			mAtbToUnqVals.put(colTokens[idx], new ArrayList<String>());
		}
	}



	/**
	 * Add training data
	 * @param colTokens the column tokens
	 */
	private void addTrainingData(String[] colTokens){
		HashMap<String,String> atbToAtbVals = new HashMap<String,String>();
		for(int idx=0; idx<colTokens.length;idx++){
			atbToAtbVals.put(mAttributes.get(idx),colTokens[idx]);
			updateAttributebUniqueVals(mAttributes.get(idx),colTokens[idx]);
		}
		mTrainingData.add(atbToAtbVals);
	}

	/**
	 * Update attributes' unique values.
	 *
	 * @param header the header
	 * @param atbVal the attributes value.
	 */
	private void updateAttributebUniqueVals(String header, String atbVal){
		List<String> unqVals=  mAtbToUnqVals.get(header);
		boolean isPresent =false;
		for(int idx=0; idx<unqVals.size();idx++){
			if(unqVals.contains(atbVal)){
				isPresent=true;
				break;
			}
		}
		if(!isPresent){
			unqVals.add(atbVal);
			mAtbToUnqVals.put(header,unqVals);	
		}
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		NaiveBayes nb = new NaiveBayes();
		String testDataPath="../Ass5-Demo/data2";
		String trainDataPath="../Ass5-Demo/data1";
		String resultFilepath = "./Results.txt";
		try {
			nb.train(trainDataPath, " ",4);
			nb.classify(testDataPath, " ");
			nb.evaluate(resultFilepath);
			
		} catch (IOException e) {
			System.out.println("Error >> Unable to find training data. "
					+ "Check path!");
		}

	}

}
