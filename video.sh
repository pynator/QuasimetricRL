#export MUJOCO_GL=osmesa

python video.py \
    --path "C:\Users\patri\Desktop\RL\gcrl\testpher\push\1730100610.9195414_test_FetchPick_her_iqe_lrA0.001_lrC0.001_seed200" \
    --name "model.pt" \
    --env_id FetchPickAndPlace-v2 \
    --episode_length 50 \
    --demo_length 10