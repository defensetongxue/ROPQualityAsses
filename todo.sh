python -u  train.py --cfg ./configs/resnet50.json --split_name 1 
python -u  train.py --cfg ./configs/resnet50.json --split_name 2 
python -u  train.py --cfg ./configs/resnet50.json --split_name 3 
python -u  train.py --cfg ./configs/resnet50.json --split_name 4

python -u  train.py --cfg ./configs/inceptionv3.json --split_name 1 
python -u  train.py --cfg ./configs/inceptionv3.json --split_name 2 
python -u  train.py --cfg ./configs/inceptionv3.json --split_name 3 
python -u  train.py --cfg ./configs/inceptionv3.json --split_name 4

python -u  train.py --cfg ./configs/vgg16.json --split_name 1 
python -u  train.py --cfg ./configs/vgg16.json --split_name 2 
python -u  train.py --cfg ./configs/vgg16.json --split_name 3 
python -u  train.py --cfg ./configs/vgg16.json --split_name 4
python ring.py

shutdown