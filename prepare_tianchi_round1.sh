save_dir="/home/xiangjianjian/dataset/tianchi/round1"

image_1_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round1_train_20190619/image_1.png"
label_1_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round1_train_20190619/image_1_label.png"

image_2_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round1_train_20190619/image_2.png"
label_2_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round1_train_20190619/image_2_label.png"

echo $save_dir

# python utils/tianchi_process.py -image_path $image_1_path -label_path $label_1_path -save_dir $save_dir
python utils/tianchi_process.py -image_path $image_2_path -label_path $label_2_path -save_dir $save_dir

python utils/tianchi_concat.py -root_dir $save_dir