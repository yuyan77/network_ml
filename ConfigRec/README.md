#### Project Structure

- data：存放原始或结果数据
  - raw_data：原始数据（未处理）
  - train_data：处理好的训练数据
  - eval_data：处理好的验证数据
  - datasets：封装模型输入的dataset构建代码
  - results：模型输出结果
- feature：数据处理代码
  - word2vec：配置向量化方法的代码
    - *bags_of_words.py：词袋模型*
  - *config_parser.py*：大规模测试数据的预处理代码
  - *data_preprocess.py*：真实配置数据的处理代码
- model：存放模型使用到的各模块
  - config_encoder
  - vm_encoder
  - prediction
  - general：通用模型模块
  - *config_recommendation.py：整体模型结构*
- checkpoints：存放训练好的模型
- scripts
- utils
- *train.py*：模型训练代码
- *evaluate.py*：模型验证
- *test.py*：使用其他数据集训练或验证模型
- *metrics.py*：模型预测结果评估
- *options.py*：模型参数
- *requirements.txt*：项目使用到的包



#### 已完成

- 整体模型v2
- 数据处理v1（bags of words模型，大规模测试产生的配置数据处理）
- 训练代码
- 验证代码（精确度）



#### 未完成

- 其他参考和对比模型（代码实现）
- 更好的word embedding模型
- 真实配置数据的处理
- 真实原始配置数据按照VM粒度，根据时序划分为多组输入（feature中部分代码）

