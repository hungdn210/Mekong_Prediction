export CUDA_VISIBLE_DEVICES=0

model_name=LSTM

python -u main.py \
  --is_training 1 \
  --station 'all' \
  --run_phase_a \
  --model $model_name \
  --learning_rate 0.01 \
  --e_layers 2 \
  --d_model 512 \
  --train_epochs 100
