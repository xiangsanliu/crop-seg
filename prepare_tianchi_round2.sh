save_dir="/home/xiangjianjian/dataset/tianchi/round2"



image_10_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round2_train_20190726/image_10.png"
label_10_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round2_train_20190726/image_10_label.png"
image_11_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round2_train_20190726/image_11.png"
label_11_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round2_train_20190726/image_11_label.png"
image_20_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round2_train_20190726/image_20.png"
label_20_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round2_train_20190726/image_20_label.png"
image_21_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round2_train_20190726/image_21.png"
label_21_path="/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/jingwei_round2_train_20190726/image_21_label.png"


# python utils/tianchi_process.py -image_path $image_10_path -label_path $label_10_path -save_dir $save_dir
# python utils/tianchi_process.py -image_path $image_11_path -label_path $label_11_path -save_dir $save_dir
# python utils/tianchi_process.py -image_path $image_20_path -label_path $label_20_path -save_dir $save_dir
# python utils/tianchi_process.py -image_path $image_21_path -label_path $label_21_path -save_dir $save_dir


python utils/tianchi_concat.py -root_dir $save_dir