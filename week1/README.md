## week 1 

## 1. 学习内容

- 环境配置，conda+IDE，强烈推荐学Linux
- Python语法
- Numpy基础，参照Numpy教程，了解Numpy的基本操作即可
- Pandas基础，参照Pandas教程，了解Pandas的基本操作、读取保存文件
- Sklearn基础，模型的调用、数据的划分、模型的验证等

## 2. 完成项目

### 2.1 iris分类

完成`iris`，项目结构

```
-iris
	-baseline.ipynb 代码样例展示
	-iris.ipynb 按照要求完成代码
```

通过这个项目，初步认识分类问题，知道数据处理、特征工程是什么东西，学会模型调用。你需要补充完成`iris.ipynb` 。

### 2.2 EEG分类

完成EEG分类，项目结构

```
- EEG
	- data 数据处理方法
		- data_merge.py 合并数据
		- data_process.py 处理数据
	- Datasets 下载的数据保存在Datasets下
	- pic 图片
	- result 结果保存目录
	- utils 工具目录，放一些函数，模型等
	- main.py 主函数
	- show.ipynb 对结果的可视化
```

通过这个项目，进一步学习数据处理和特征工程，你需要补充完成`utils/utils.py` 下的`test_model`函数，使用各种模型并调节参数。

