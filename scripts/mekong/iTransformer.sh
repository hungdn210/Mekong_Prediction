export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u main.py \
  --is_training 1 \
  --station 'all' \
  --run_phase_a \
  --model $model_name \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --d_model 128 \
  --d_ff 512 \
  --n_heads 2 \
  --e_layers 2 \
  --d_layers 1 \
  --verbose 0
