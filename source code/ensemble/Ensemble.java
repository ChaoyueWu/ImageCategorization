package thu.coursr.ictask.ensemble;

public class Ensemble {

	public static void main(String[] args) {
		String resnet50Path = "1.list";
		String v2Path = "2.list";
		String alexnetPath = "3.list";
		String savePath = "result.list";
		
		//读取label文件
		Calculate cal = new Calculate();
		cal.calculate(resnet50Path, v2Path, alexnetPath,savePath);
	}

}
