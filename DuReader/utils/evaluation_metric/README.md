#MRC2018 Evaluation Metrics with Bounce
##Metric Details
Please refer to 
>MRC2018 Metrics.docx

##How to run the script
Run the demo data:
>python mrc_eval.py demo_data/pred.json demo_data/ref.json

In this evaluation we set alpha=1.0, beta=1.0, gamma=1.2 by default.

### pred.json 格式
```
{
	"yesno_answers": [],
	"question": "丹霞山门票多少钱2017",
	"question_type": "ENTITY",
	"answers": ["成人票200元，预定价140元。1、儿童身高1.2米以下免门票，1.2-1.5米儿童半价。六一儿童节当日，身高不足1.5米儿童给予免景区门票，教师节当日，教师凭本人有效教师资格证免景区门票。2、记者（凭本人新闻总署颁发的记者证）、残疾人（凭本人残"],
	"question_id": 184936
}
```


### ref.json
```
{
	"entity_answers": [
		[]
	],
	"yesno_answers": [],
	"question": "丹霞山门票多少钱2017",
	"question_type": "ENTITY",
	"answers": ["成人票200元，预定价140元。1、儿童身高1.2米以下免门票，1.2-1.5米儿童半价。六一儿童节当日，身高不足1.5米儿童给予免景区门票，教师节当日，教师凭本人有效教师资格证免景区门票。2、记者（凭本人新闻总署颁发的记者证）、残疾人（凭本人残疾证）、70周岁以上老年人（凭本人身份证或老年证）免景区门票。免票人员必须购买3元/人的保险费方可进山。3、60-69周岁老年人（凭本人身份证或老年证）、现役军人（凭本人现役军人证）、学生（凭本人有效学生证、成人教育等非全日制学生除外）门票半价。"],
	"question_id": 184936,
	"source": "search"
}
```


