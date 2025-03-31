export CUDA_VISIBLE_DEVICES=0

model_name = DLinear

stations=(
  'Kompong Cham' \
  'Chaktomuk' \
  'Vientiane KM4' \
  'Ban Pak Kanhoung' \
  'Ban Na Luang' \
  'Chiang Saen' \
  'Ban Huai Yano Mai' \
  'Ban Tha Ton' \
  'Ban Tha Mai Liam' \
  'Ban Pak Huai' \
  'Yasothon' \
  'Ban Chot' \
  'Ban Nong Kiang' \
  'Ban Tad Ton' \
  'Ban Huai Khayuong' \
  'Cau 14 (Buon Bur)' \
  'Stung Treng' \
  'Pakse' \
  'Ban Kengdone' \
  'Chiang Khan' \
  'Nong Khai' \
  'Nakhon Phanom' \
  'Mukdahan' \
  'Khong Chiam' \
  'Kontum' \
  'Duc Xuyen' \
)

seq_len=365
pred_len=30

for station_i in ${!stations[@]}
do
  station_name=${stations[station_i]}
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path './dataset/Filled_Gaps_Mekong_Database/Discharge.Daily/' \
    --data_path "${station_name}.csv" \
    --model_id "${station_name}_${seq_len}_${pred_len}" \
    --model $model_name \
    --data MeKong \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --learning_rate 0.01 \
    --inverse 1 \
    --des 'Exp' \
    --itr 1
done