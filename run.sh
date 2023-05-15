# 生成数据集配置
uav_number=2;
user_number=10;
number_of_train_data=15000;
number_of_test_data=5000;
overtime_penalty=100
# 100 -> 1/4的energy cost
# 200 -> 1/2的energy cost

# ---------------------
# 模型配置
hidden_dim=512;
drop_out=0.1;
reg_better_sol_number=15;
__epoch_num=10;
# train_iter=`expr $number_of_train_data \* $__train_model_every_k_step \* $__epoch_num`;
train_iter=`expr $number_of_train_data \* $__epoch_num`;
# train_iter=35000                                # 用于调试

# ------------------------------------
# 流程配置
generate_dataset=false
test_no_reg_method=false

# 生成数据
if ($generate_dataset = true)
then
    echo "[System] Generating training and testing dataset ..."
    python dataclass.py --uavNumber $uav_number \
                        --userNumber $user_number \
                        --penalty $overtime_penalty \
                        --number_of_train_data $number_of_train_data \
                        --number_of_test_dat $number_of_test_data \
                        # --using_random_sol 
fi
# -----------------------------------------------------
# # 模型1MLP:
model_name="mlp"
# 训练中生成更好的解
echo "[System] Training model ${model_name}, Generate Better Solution During Training = True ..."
python train.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number \
                --batch_size 256 \
                --hidden_dim $hidden_dim \
                --num_of_iter $train_iter \
                --drop_out $drop_out \
                --reg_better_sol_k $reg_better_sol_number \
                --reg_better_sol > "./Log/[TRAIN_LOG]_${model_name}_train_log".txt

echo "[System] Perform Testing ..."
python test_compare_diff_methods.py \
                --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number > "./Log/[TEST_LOG]_${model_name}".txt

if ($test_no_reg_method = true)
then
    # 训练中不生成更好的解
    echo "[System] Training model ${model_name}, Generate Better Solution During Training = False ..."
    python train.py --nnModel $model_name \
                    --uavNumber $uav_number \
                    --userNumber $user_number \
                    --batch_size 256 \
                    --hidden_dim $hidden_dim \
                    --num_of_iter $train_iter \
                    --drop_out $drop_out > "./Log/[TRAIN_LOG]_${model_name}_without_reg_better_solution_train_log".txt

    echo "[System] Perform Testing ..."
    python test_compare_diff_methods.py \
                    --test_NN_only \
                    --nnModel $model_name \
                    --uavNumber $uav_number \
                    --userNumber $user_number > "./Log/[TEST_LOG]_${model_name}_without_reg_better_solution".txt
fi

# -----------------------------------------------------
# 模型2LSTM:
model_name="LSTM"
# 训练中生成更好的解
echo "[System] Training model ${model_name}, Generate Better Solution During Training = True ..."
python train.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number \
                --batch_size 256 \
                --hidden_dim $hidden_dim \
                --num_of_iter $train_iter \
                --drop_out $drop_out \
                --reg_better_sol_k $reg_better_sol_number \
                --reg_better_sol > "./Log/[TRAIN_LOG]_${model_name}_train_log".txt

echo "[System] Perform Testing ..."
python test_compare_diff_methods.py \
                --test_NN_only \
                --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number > "./Log/[TEST_LOG]_${model_name}".txt

if ($test_no_reg_method = true)
then
    # 训练中不生成更好的解
    echo "[System] Training model ${model_name}, Generate Better Solution During Training = False ..."
    python train.py --nnModel $model_name \
                    --uavNumber $uav_number \
                    --userNumber $user_number \
                    --batch_size 256 \
                    --hidden_dim $hidden_dim \
                    --num_of_iter $train_iter \
                    --drop_out $drop_out > "./Log/[TRAIN_LOG]_${model_name}_without_reg_better_solution_train_log".txt

    echo "[System] Perform Testing ..."
    python test_compare_diff_methods.py \
                    --test_NN_only \
                    --nnModel $model_name \
                    --uavNumber $uav_number \
                    --userNumber $user_number > "./Log/[TEST_LOG]_${model_name}_without_reg_better_solution".txt
fi

# -----------------------------------------------------
# 模型3 LSTM + ATTENTION
model_name="LSTM_ATT"
# 训练中生成更好的解
echo "[System] Training model ${model_name}, Generate Better Solution During Training = True ..."
python train.py --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number \
                --batch_size 256 \
                --hidden_dim $hidden_dim \
                --num_of_iter $train_iter \
                --drop_out $drop_out \
                --reg_better_sol_k $reg_better_sol_number \
                --reg_better_sol > "./Log/[TRAIN_LOG]_${model_name}_train_log".txt

echo "[System] Perform Testing ..."
python test_compare_diff_methods.py \
                --test_NN_only \
                --nnModel $model_name \
                --uavNumber $uav_number \
                --userNumber $user_number > "./Log/[TEST_LOG]_${model_name}".txt

if ($test_no_reg_method = true)
then
    # 训练中不生成更好的解
    echo "[System] Training model ${model_name}, Generate Better Solution During Training = False ..."
    python train.py --nnModel $model_name \
                    --uavNumber $uav_number \
                    --userNumber $user_number \
                    --batch_size 256 \
                    --hidden_dim $hidden_dim \
                    --num_of_iter $train_iter \
                    --drop_out $drop_out > "./Log/[TRAIN_LOG]_${model_name}_without_reg_better_solution_train_log".txt

    echo "[System] Perform Testing ..."
    python test_compare_diff_methods.py \
                    --test_NN_only \
                    --nnModel $model_name \
                    --uavNumber $uav_number \
                    --userNumber $user_number > "./Log/[TEST_LOG]_${model_name}_without_reg_better_solution".txt
fi