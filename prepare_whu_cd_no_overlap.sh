save_dir="dataset/whu-cd"

test_image_2012="dataset/whu-cd-arch/2012_test_image.tif"
test_label_2012="dataset/whu-cd-arch/2012_test.tif"

train_image_2012="dataset/whu-cd-arch/2012_train_image.tif"
train_label_2012="dataset/whu-cd-arch/2012_train.tif"

test_image_2016="dataset/whu-cd-arch/2016_test_image.tif"
test_label_2016="dataset/whu-cd-arch/2016_test.tif"

train_image_2016="dataset/whu-cd-arch/2016_train_image.tif"
train_label_2016="dataset/whu-cd-arch/2016_train.tif"

change_label_test="dataset/whu-cd-arch/change_label_test.tif"
change_label_train="dataset/whu-cd-arch/change_label_train.tif"

python utils/whu_cd_process_no_overlap.py -image_path $train_image_2012 -save_dir $save_dir -type image -suffix train_1
python utils/whu_cd_process_no_overlap.py -image_path $train_image_2016 -save_dir $save_dir -type image -suffix train_2

python utils/whu_cd_process_no_overlap.py -image_path $test_image_2012 -save_dir $save_dir -type image -suffix test_1
python utils/whu_cd_process_no_overlap.py -image_path $test_image_2016 -save_dir $save_dir -type image -suffix test_2



python utils/whu_cd_process_no_overlap.py -image_path $train_label_2012 -save_dir $save_dir -type label -suffix train_1
python utils/whu_cd_process_no_overlap.py -image_path $train_label_2016 -save_dir $save_dir -type label -suffix train_2

python utils/whu_cd_process_no_overlap.py -image_path $test_label_2012 -save_dir $save_dir -type label -suffix test_1
python utils/whu_cd_process_no_overlap.py -image_path $test_label_2016 -save_dir $save_dir -type label -suffix test_2
python utils/whu_cd_process_no_overlap.py -image_path $change_label_test -save_dir $save_dir -type label -suffix test_change
python utils/whu_cd_process_no_overlap.py -image_path $change_label_train -save_dir $save_dir -type label -suffix train_change


# python utils/tianchi_concat.py -root_dir $save_dir
# python utils/tianchi_concat_random.py -root_dir $save_dir