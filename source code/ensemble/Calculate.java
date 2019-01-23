package thu.coursr.ictask.ensemble;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class Calculate {

	public void calculate(String p1,String p2,String p3,String savePath) {
		Map<String,String> map1 = readFile(p1);
		Map<String,String> map2 = readFile(p2);
		Map<String,String> map3 = readFile(p3);
		Map<String,String> result = new HashMap<String,String>(); 
		Set<String> idSet = map1.keySet();
		
		//遍历
		String l1,l2,l3,resultLabel;
		for(String id:idSet) {
			l1 = map1.get(id);
			l2 = map2.get(id);
			l3 = map3.get(id);
			
			if(l1.equals(l2)&&l1.equals(l3)) {	//1=2=3
				resultLabel = l1;
			}else if(l1.equals(l2)&&(!l1.equals(l3))) {	//ne3
				resultLabel = l1;
			}else if(l3.equals(l2)&&(!l1.equals(l3))) {	//ne1
				resultLabel = l3;
			}else if(l1.equals(l3)&&(!l1.equals(l2))) {	//ne2
				resultLabel = l1;
			}else{
				resultLabel = l1;
			}
			result.put(id, resultLabel);
		}
		 
		 
		//write
		try {
			FileWriter fw = new FileWriter(new File(savePath)) ;
			for (Map.Entry<String, String> entry : result.entrySet()) { 
				  fw.write(entry.getKey());
				  fw.write(' ');
				  fw.write(entry.getValue());
				  fw.write('\n');
			}
			fw.flush();
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	private Map<String,String> readFile(String path) {
		Map<String,String> map = new HashMap<String,String>(); 
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(path)));
			String line = "";
			String[] arr;
			while((line=br.readLine())!=null) {
				arr = line.split(" ");
				map.put(arr[0], arr[1]);
			}
			br.close();
			return map;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return map;
	}

}
