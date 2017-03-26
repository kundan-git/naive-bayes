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
	private boolean KEEP_DUPLICATES = true;//false;

	/** The unique attributes. */
	private List<String> mAttributes = new ArrayList<String>();

	/** The attributes to unique values map. */
	private HashMap<String, List<String>> mAtbToUnqVals= new HashMap<String,List<String>>();

	/** The training data read from file and pre-processed.*/
	private List<HashMap<String,String> > mTrainingData = new ArrayList<HashMap<String, String>>();

	private int mTargetIdx=-1;


	
	public void train(String trainDataPath, String delimiter, int targetIdx) throws IOException {
		if(!(delimiter.equals(SPACE) || delimiter.equals(COMMA) )){
			System.out.println("Error >> Invalid delimiter specified.");
			return;
		}
		
		mTargetIdx = targetIdx;
		
		/* Read training data from file.*/
		readTrainingData(trainDataPath,delimiter);

		/* Remove spurious and duplicate data*/
		preProcessTrainingData();



	}


	private void printTrainingData(int dataIdx){
		String data="";
		HashMap<String, String> vals= mTrainingData.get(dataIdx);
		for(String val: vals.keySet()){
			data= data+" "+vals.get(val);
		}
		System.out.println(data);
	}
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
					System.out.println("Same attributes:"+idx+":"+idx1);
					printTrainingData(idx);printTrainingData(idx1);System.out.println("\n");
					
					/* If target value of oneAtbToVal and atbToVal are different, */
					/* drop both. If same, drop if duplicates are not allowed.    */
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
	private void readTrainingData(String trainDataPath, String delimiter) throws IOException{
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
		String trainDataPath="../Ass5-Demo/data2";
		try {
			nb.train(trainDataPath, " ",4);
		} catch (IOException e) {
			System.out.println("Error >> Unable to find training data. Check path!");
		}

	}

}
