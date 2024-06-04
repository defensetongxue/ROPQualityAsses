<<<<<<< HEAD
python train.py  --cfg ./configs/resnet50.json --angle_type 0 --split_name 1
=======
# python train.py  --cfg ./configs/resnet50.json --angle_type 0 --split_name 1
>>>>>>> 1059803015fdc7242d6403ad5c0efbc6b914dd73
python train.py  --cfg ./configs/resnet50.json --angle_type 0 --split_name 2
python train.py  --cfg ./configs/resnet50.json --angle_type 0 --split_name 3
python train.py  --cfg ./configs/resnet50.json --angle_type 0 --split_name 4

# python test.py  --cfg ./configs/resnet50.json --angle_type 0 --split_name 1
python test.py  --cfg ./configs/resnet50.json --angle_type 0 --split_name 2
python test.py  --cfg ./configs/resnet50.json --angle_type 0 --split_name 3
python test.py  --cfg ./configs/resnet50.json --angle_type 0 --split_name 4
python ring.py
shutdown