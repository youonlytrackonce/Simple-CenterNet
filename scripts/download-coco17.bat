mkdir dataset
cd dataset
mkdir coco17
cd coco17
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

tar xf train2017.zip
tar xf val2017.zip
tar xf test2017.zip
tar xf annotations_trainval2017.zip
