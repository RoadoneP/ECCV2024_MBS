cd data_root

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip
wget http://cs.jhu.edu/~cxliu/data/list.zip
rm VOCtrainval_11-May-2012.tar

unzip SegmentationClassAug.zip
unzip SegmentationClassAug_Visualization.zip
unzip list.zip

mv SegmentationClassAug ./VOCdevkit/VOC2012/
mv SegmentationClassAug_Visualization ./VOCdevkit/VOC2012/
mv list ./VOCdevkit/VOC2012/

rm list.zip
rm SegmentationClassAug.zip
rm SegmentationClassAug_Visualization.zip
