uav_number=3;
user_number=10;
number_of_train_data=15000;
number_of_test_data=5000;

# ---------------------
# 模型配置
hidden_dim=512;
drop_out=0.20;
reg_better_sol_number=20;
train_iter=`expr $number_of_train_data \* 10`

# 生成数据
python dataclass.py --uavNumber $uav_number \
                    --userNumber $user_number \
                    --number_of_train_data $number_of_train_data \
                    --number_of_test_dat $number_of_test_data \
                    # --using_random_sol 

# -----------------------------------------------------
model_name="mlp"
python train.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number \
                --batch_size 256 \
                --hidden_dim $hidden_dim \
                --num_of_iter $train_iter \
                --drop_out $drop_out \
                --reg_better_sol_k $reg_better_sol_number \
                --reg_better_sol

python test_compare_diff_methods.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number > $model_name.txt

# -----------------------------------------------------
model_name="LSTM"
python train.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number \
                --batch_size 256 \
                --hidden_dim $hidden_dim \
                --num_of_iter $train_iter \
                --drop_out $drop_out \
                --reg_better_sol_k $reg_better_sol_number \
                --reg_better_sol

python test_compare_diff_methods.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number > $model_name.txt

# -----------------------------------------------------
# LSTM ATTN
model_name="LSTM_ATT"
python train.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number \
                --batch_size 256 \
                --hidden_dim $hidden_dim \
                --num_of_iter $train_iter \
                --drop_out $drop_out \
                --reg_better_sol_k $reg_better_sol_number \
                --reg_better_sol

python test_compare_diff_methods.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number > $model_name.txt
