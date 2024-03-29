﻿### 此内容公开 ###


一、任务描述
    （1）设计和实现分类系统，完成对文本的情感分类任务，这里包含三种情感：中性，积极和消极。
         程序语言、框架、学习方法不限，可使用外部语料，不可使用已有的情感分析或文本分类库。

    （2）“train_data”目录下提供8,606条训练数据，这里每个数据集采用json格式进行存储；
         文件为“UTF-8”编码，数据以json格式存储，格式如下：
            {
                "id": 11,
                "content": "武汉加油！中国加油！安徽加油！",
                "label": "positive"
            }
         “id”表示数据标号，“content”表示文本内容，“label”表示情感类别。

    （3）“eval_data”目录下面文件“eval.json”为开发集，一共包括2,000条数据，用于模型调参，其数据格式和训练集一致。

    （4）“test_data”目录下是文件“test.json”，包含3,000条未知类别的测试数据，使用训练好的模型对其进行预测。
        注：测试数据将于6月1号发放，在此之前，请同学们使用开发集进行模型参数调优。
        文件为“UTF-8”编码，数据以json格式存储，格式如下：
            {
                "id": 10,
                "content": "#新型冠状病毒肺炎纳入法定传染病#"
            },
         “id”表示数据标号，“content”表示文本内容。

         **重要！！！**
         对测试数据进行预测，为了表示方便，规定如下映射关系：
         中性用0表示
         积极用1表示
         消极用2表示


二、提交说明
    （1）每人提交一份结果，结果是一个“.csv”格式的数据文件（以逗号“,”为分隔符，这里的是英文格式下的逗号），以学号姓名命名。（即：学号-姓名.csv，例如test_data目录下的21S123456-张三.csv，但是这个文件的结果都是随机生成的，所以不要提交这个文件）
    （2）“.csv”文件应为3,000行2列，每一行是一条测试数据的预测结果。
         第一列是测试数据id（这里的id的值要和test.json中的id值一一对应），第二列是情感极性预测结果（0-中性，1-积极，2-消极）。
         （注：以0、1、2表示预测结果，切勿用“中性积极消极”等其它字符）
    （3）提交的“学号-姓名.csv”文件一定是一个3,000行2列，并以逗号“,”（这里的是英文格式下的逗号）为分隔符的数据文件，不符合要求的提交得分暂时记为0。
    （4）要求提交的文件为‘UTF-8’编码，不要提交‘GBK’及其他编码的文件。
    （5）评分标准：以宏平均F1（macro-averaged F1-score）作为评分标准。


  
