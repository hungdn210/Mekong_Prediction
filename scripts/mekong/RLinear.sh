export CUDA_VISIBLE_DEVICES=0

model_name=RLinear

python -u main.py \
  --is_training 1 \
  --station 'all' \
  --run_phase_a \
  --model $model_name \
  --learning_rate 0.01 \
  --train_epochs 100 \
  --verbose 1
