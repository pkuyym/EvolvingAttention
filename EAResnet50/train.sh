PYTHONPATH=. python -u official/resnet/imagenet_main.py \
    --data_format='channels_first' \
    --data_dir=${YOUR_IMAGENET_TFRECORDS_DIR} \
    --model_dir=${OUTPUT_DIR} \
    --batch_size=256 \
    --resnet_size=50 \
    --train_epochs=100 \
    --num_gpus=8 \
