# DES-PSP
 Double Encoder Seq2seq-Based Presidential Conceptual Stock Price Prediction model

**NOTIFICATION:**

_Note : Model need interface._

*Note: The README Still need a further revise and implements.* 

*Note: Reproduce paper list is need to be determined.*

*Note: The test parameter need to be appointed.*

__Update:__

* *Refactored the whole repo. 10-25-2023*
* *A argparse interface is released. 09-015-2023*

**TO-DO:**

1. Ablation Study.
2. Need to compare the proposed DES-PSP model with existing state-of-the-art methods in the field of stock market prediction.
3. Need more comprehensive analysis of the model's sensitivity to different parameter settings.
4. Potentially combine novelty methods (After well-done baseline).
5. Verifying the hypothesis of the exists stock which benefits from the election of a specific presidential candidate.
6. Find background news to support the election of conceptual stocks.
7. State the generalizability.

## DataSet

The full dataset : ./data/raw/raw/All_Data.csv

## Use

* To **train** the Seq2Seq baseline model :

```shell
python main_baseline.py --model=seq2seq_lstm
```



## Results

### Price

| Model          | MSE        | RMSE       | MAE        | ADE        | FDE        |
| -------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| bigru          | 0.8492     | 0.9215     | 0.5851     | 0.5851     | 0.7506     |
| bilstm         | 0.7023     | 0.8380     | 0.5322     | 0.5322     | 0.6691     |
| gru            | 0.8440     | 0.9187     | 0.5756     | 0.5756     | 0.7729     |
| lstm           | 0.6596     | 0.8122     | 0.5215     | 0.5215     | 0.6862     |
| seq2seq_bigru  | 0.9191     | 0.9587     | 0.6043     | 0.6043     | 0.8206     |
| seq2seq_bilstm | 0.7031     | 0.8385     | 0.5299     | 0.5299     | **0.6731** |
| seq2seq_gru    | 0.7780     | 0.8821     | 0.5509     | 0.5509     | 0.6869     |
| seq2seq_lstm   | 0.6764     | 0.8224     | 0.5243     | 0.5243     | 0.6961     |
| DES-PSP        | **0.6372** | **0.7982** | **0.5201** | **0.5201** | 0.7452     |

### Movement

| Model          | Accuracy   | F1         | MCC        |
| -------------- | ---------- | ---------- | ---------- |
| bigru          | 0.8268     | 0.9041     | 0.0988     |
| bilstm         | 0.8317     | 0.9079     | 0.0393     |
| gru            | **0.8329** | **0.9088** | 0.0085     |
| lstm           | 0.8324     | 0.9085     | 0.0014     |
| seq2seq_bigru  | 0.8085     | 0.8909     | 0.1347     |
| seq2seq_bilstm | 0.8313     | 0.9074     | 0.0715     |
| seq2seq_gru    | 0.8304     | 0.9070     | 0.0498     |
| seq2seq_lstm   | 0.8327     | 0.9087     | 0.0198     |
| DES-PSP        | 0.8293     | 0.9046     | **0.1628** |





## Citation

Please cite our paper if you use it in your work:

```shell
@inproceedings{,
   title={{}: },
   author={},
   booktitle={},
   year={}
}
```
