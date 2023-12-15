3个时间段

[08 12] [12 16] [16 20]

12, 16, 20年概念股预测：只预测挑选的股票(baseline训练挑选,预测挑选; DESPSP训练挑选预测挑选)

- 任务1：movement预测 t+20 
  - ACC, F1, MCC
  
- 任务2： value预测 t+20
  - RMSE MAE FDE (ADE)
  
- 任务3： 选股 Concept Identification Accuracy

  - ACC, F1, MCC, Recall

    

20年概念股预测：主实验:预测全量的股票(baseline训练全量预测挑选, despsp训练挑选预测挑选)

- 任务1：movement预测 t+20
  - ACC, F1, MCC
- 任务2： value预测 t+20
  - RMSE MAE FDE (ADE)
- 任务3： 选股 Concept Identification Accuracy
  - ACC, F1, MCC



消融实验(只用16 20)

- 打乱分组
- 预测多少天t+n [12 14 16 18 20]
- fr有用吗 [没有用]

- LSTM CNN 层数(不一定)
- encoder decoder 换成其他RNN(不一定)