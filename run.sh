uav_number=3
user_number=10
number_of_train_data=3000
number_of_test_data=500
# ---------------------
# 模型配置
# model_name="mlp"
# model_name="LSTM"
model_name="LSTM_ATT"
hidden_dim=256
train_iter=12000

# python dataclass.py --uavNumber $uav_number \
#                     --userNumber $user_number \
#                     --number_of_train_data $number_of_train_data \
#                     --number_of_test_dat $number_of_test_data \
#                     # --using_random_sol 

python train.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number \
                --batch_size 256 \
                --hidden_dim $hidden_dim \
                --num_of_iter $train_iter \
                --drop_out 0.3 \
                # --reg_better_sol

python test_compare_diff_methods.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number \