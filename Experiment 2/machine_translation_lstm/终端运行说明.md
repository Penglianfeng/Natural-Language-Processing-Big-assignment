# Terminal instructions

From the root of the current workspace, run:

```bash
python machine_translation_lstm\seq2seq_lstm_attention.py
` ` `

The program does this automatically:

1. Construct English-Chinese parallel corpus;
2. Train a normal multi-layer LSTM Seq2Seq model;
Train multi-layer LSTM Seq2Seq model with attention mechanism; 3.
4. Output the sample translation results;
5. Save loss curve, attention heat map, and screenshots of experimental results.

The results are stored in:

```text
machine_translation_lstm\results\
` ` `

Main files:

- 'result_screenshot.png' : you can paste a screenshot of the running results directly into the experiment report;
- 'loss_curve.png' : train loss curve;
- attention_heatmap.png: attention heatmap;
- 'predictions.txt' : terminal output translation results;
- 'metrics.json' : experimental metrics.
