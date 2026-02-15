# nohup python -u test.py > ./logs/bbi-d_gray_pg0.01+0.0002.log 2>&1 &
#   ["pg0.01+0.0002", "pg0.01+0.02", "pg0.05+0.02"]
#### BBI-Denoiser
## synthetic
# gray
python train_est_gray.py --gpu_devices 0 --noisetype pg0.01+0.0002 --save_model_path ./experiments/baselines --log_name est_bbi_gray_pg0.01+0.0002
python train_vmapest_joint_gray.py --gpu_devices 0 --noisetype pg0.01+0.0002 --save_model_path ./experiments/baselines --log_name vmap_est_bbi_gray_pg0.01+0.0002_13rf21 --Lambda1 1 --Lambda2 3 --increase_ratio 21.0

# rgb
python train_est_rgb.py --gpu_devices 0 --noisetype pg0.01+0.0002 --save_model_path ./experiments/baselines --log_name est_bbi_rgb_pg0.01+0.0002
python train_vmapest_joint_rgb.py --gpu_devices 0 --noisetype pg0.01+0.0002 --save_model_path ./experiments/baselines --log_name vmap_est_bbi_rgb_pg0.01+0.0002_13rf21 --Lambda1 1 --Lambda2 3 --increase_ratio 21.0

# sRGB
python train_est_sRGB.py --gpu_devices 0 --noisetype pg0.01+0.0002 --save_model_path ./experiments/baselines --log_name est_bbi_sRGB_pg0.01+0.0002
python train_vmapest_joint_sRGB.py --gpu_devices 0 --noisetype pg0.01+0.0002 --save_model_path ./experiments/baselines --log_name vmap_est_bbi_sRGB_pg0.01+0.0002_13rf21 --Lambda1 1 --Lambda2 3 --increase_ratio 21.0

## real
# SIDD
python train_est_sidd.py --gpu_devices 0 --lr 1e-4 --batchsize 1 --patchsize 256 --save_model_path ./experiments/est_sidd --log_name est_bbi_unet_raw
python train_vmapest_joint_sidd.py --gpu_devices 0 --save_model_path ./experiments/vmap_est_sidd --log_name vmap_est_bbi_unet_raw_13rf21 --Lambda1 1 --Lambda2 3 --increase_ratio 21.0

# FMD: ['Confocal_FISH','Confocal_MICE','TwoPhoton_MICE']
python train_est_fmdd.py --gpu_devices 0 --lr 1e-4 --batchsize 1 --patchsize 256 --subfold Confocal_FISH --save_model_path ./experiments/est_fmdd --log_name Confocal_FISH_est_bbi_unet_fmdd
python train_vmapest_joint_fmdd.py --gpu_devices 0 --lr 1e-3 --batchsize 4 --subfold Confocal_FISH --save_model_path ./experiments/vmap_est_fmdd --log_name Confocal_FISH_vmap_est_bbi_unet_fmdd_13rf21 --Lambda1 1 --Lambda2 3 --increase_ratio 21.0


# sepG
# fmdd
nohup python -u train_cov_joint_fmdd_sepG.py --gpu_devices 3 --lr 4e-4 --batchsize 4 --subfold Confocal_MICE --save_model_path ./experiments/baselines/fmdd --log_name Confocal_MICE_fmdd_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/Confocal_MICE_fmdd_13rf11_west0.0_sepG.log 2>&1 &
# sidd
nohup python -u train_cov_joint_sidd_sepG.py --gpu_devices 3 --lr 1e-4 --batchsize 4 --save_model_path ./experiments/baselines/sidd --log_name sidd_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/sidd_13rf11_west0.0_sepG.log 2>&1 &
# gray
nohup python -u train_cov_joint_gray_sepG.py --gpu_devices 3 --noisetype pg0.01+0.0002 --lr 1e-3 --batchsize 4 --save_model_path ./experiments/baselines/gray --log_name gray_pg0.01+0.0002_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/gray_pg0.01+0.0002_13rf11_west0.0_sepG.log 2>&1 &
nohup python -u train_cov_joint_gray_sepG.py --gpu_devices 3 --noisetype pg0.01+0.02 --lr 1e-3 --batchsize 4 --save_model_path ./experiments/baselines/gray --log_name gray_pg0.01+0.02_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/gray_pg0.01+0.02_13rf11_west0.0_sepG.log 2>&1 &
nohup python -u train_cov_joint_gray_sepG.py --gpu_devices 3 --noisetype pg0.05+0.02 --lr 1e-3 --batchsize 4 --save_model_path ./experiments/baselines/gray --log_name gray_pg0.05+0.02_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/gray_pg0.05+0.02_13rf11_west0.0_sepG.log 2>&1 &
# rgb
nohup python -u train_cov_joint_rgb_sepG.py --gpu_devices 3 --noisetype pg0.01+0.0002 --lr 1e-3 --batchsize 4 --save_model_path ./experiments/baselines/rgb --log_name rgb_pg0.01+0.0002_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/rgb_pg0.01+0.0002_13rf11_west0.0_sepG.log 2>&1 &
nohup python -u train_cov_joint_rgb_sepG.py --gpu_devices 3 --noisetype pg0.01+0.02 --lr 1e-3 --batchsize 4 --save_model_path ./experiments/baselines/rgb --log_name rgb_pg0.01+0.02_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/rgb_pg0.01+0.02_13rf11_west0.0_sepG.log 2>&1 &
nohup python -u train_cov_joint_rgb_sepG.py --gpu_devices 3 --noisetype pg0.05+0.02 --lr 1e-3 --batchsize 4 --save_model_path ./experiments/baselines/rgb --log_name rgb_pg0.05+0.02_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/rgb_pg0.05+0.02_13rf11_west0.0_sepG.log 2>&1 &
# sRGB
nohup python -u train_cov_joint_sRGB_sepG.py --gpu_devices 0 --noisetype pg0.01+0.0002 --lr 3e-4 --batchsize 4 --save_model_path ./experiments/baselines/sRGB --log_name sRGB_pg0.01+0.0002_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/sRGB_pg0.01+0.0002_13rf11_west0.0_sepG.log 2>&1 &
nohup python -u train_cov_joint_sRGB_sepG.py --gpu_devices 0 --noisetype pg0.01+0.02 --lr 3e-4 --batchsize 4 --save_model_path ./experiments/baselines/sRGB --log_name sRGB_pg0.01+0.02_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/sRGB_pg0.01+0.02_13rf11_west0.0_sepG.log 2>&1 &
nohup python -u train_cov_joint_sRGB_sepG.py --gpu_devices 0 --noisetype pg0.05+0.02 --lr 3e-4 --batchsize 4 --save_model_path ./experiments/baselines/sRGB --log_name sRGB_pg0.05+0.02_13r11_west0.0_sepG --w_est 0.0 --Lambda2 3.0 --increase_ratio 11.0 > ./logs/sRGB_pg0.05+0.02_13rf11_west0.0_sepG.log 2>&1 &

# mix sRGB
nohup python -u train_cov_joint_sRGB_sepnoise_mix.py --gpu_devices 2 --noisetype pg0.01+0.02 --lr 3e-4 --batchsize 4 --save_model_path ./experiments/baselines/sRGB --log_name sRGB_pg0.01+0.02_11r5_west0.0_sepnoise_mixwod --w_est 0.0 --Lambda2 1.0 --increase_ratio 5.0 > ./logs/sRGB_pg0.01+0.02_11rf5_west0.0_sepnoise_mixwod.log 2>&1 &

# pg0.01+0.0002, pg0.01+0.02, pg0.05+0.02
# test mode
python test_b2s_rgb.py --noisetype pg0.01+0.0002 --checkpoint ./experiments/ablations/rgb/rgb_pg0.01+0.0002_13r3_west0.0_sepG/2022-01-26-12-55/models/epoch_model_067.pth  --gpu_devices 0 --save_test_path ./test_b2s_rgb  --log_name  b2s_rgb_unet_pg0.01+0.0002_13r11_west0