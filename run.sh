uav_number=3
user_number=10
number_of_train_data=100
number_of_test_data=50
# ---------------------
# 模型配置
model_name="mlp"
# model_name="LSTM"
# model_name="LSTM_ATT"
hidden_dim=32
train_iter=2000

# python dataclass.py --uavNumber $uav_number \
#                     --userNumber $user_number \
#                     --number_of_train_data $number_of_train_data \
#                     --number_of_test_dat $number_of_test_data \
#                    # --using_random_sol 

python train.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number \
                --batch_size 256 \
                --hidden_dim $hidden_dim \
                --num_of_iter $train_iter \
                --drop_out 0.3

python test_compare_diff_methods.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number
                --hidden_dim $hidden_dim \