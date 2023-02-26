github 一共有以下三种运行方式
train中都有一句话 wandb.init(project="CrossPoint", name=args.exp_name) 启用wandb帮助监控
文件一train_crosspoint.py： （有很多没用的code留着没删）
# Traning CrossPoint for classification
python train_crosspoint.py --model dgcnn --epochs 100 --lr 0.001 --exp_name crosspoint_dgcnn_cls --batch_size 20 --print_freq 200 --k 15
# Training CrossPoint for part-segmentation
python train_crosspoint.py --model dgcnn_seg --epochs 100 --lr 0.001 --exp_name crosspoint_dgcnn_seg --batch_size 20 --print_freq 200 --k 15
前两种在训练网络给cls partseg子任务使用 两种网络
train_loader = DataLoader(ShapeNetRender(transform, n_imgs = 2), num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
在./datasets/data.py中有ShapeNetRender类 提到load_shapenet_data函数 用的是./data/ShapeNet文件夹下的全部
Testing的时候训练sklearn.svm.SVC （默认的核是RBF大名鼎鼎的高斯径向基函数 但testing训的是线性核 C=0.1）
ModelNet40SVM做SVM的train和test 
train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=128, shuffle=True)
test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=128, shuffle=True)



文件二train_partseg.py：
# Fine-tuning for part-segmentation
python train_partseg.py --exp_name dgcnn_partseg --pretrained_path ./models/dgcnn_partseg_best.pth --batch_size 8 --k 40 --test_batch_size 8 --epochs 300
后一种 利用partseg训出来最好的 不过这里的路径下的best是从github上下的 默认train模式（eval=False）
不过train_partseg.py也有test模式，test模式的model_path没设置。
train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice)
train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice), num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
在./datasets/shapenet_part.py里有ShapeNetPart类，有个load_data_partseg函数 用的是./data/shapenet_part_seg_hdf5_data下的.h5文件

疑问：cls没有test吗？ 为什么需要单独的train_partseg.py
疑问：k20 40有什么关系？
k用在dgcnn.py中的get_graph_feature()函数 很重要决定DGCNN中Graph的邻居个数
疑问：为什么partseg的pretrain也是Linear accuracy没有IoU？
pretrain没有分割的数据，因此
疑问：cls和partseg网络结构有何不同即DGCNN和DGCNN_partseg的区别？
已解决：看DGCNN文章，结构与PointNet类似，但是分类网络只用全局特征，分割网络需要局部拼接全局，最终输出维度也不同。


python train_crosspoint_update.py --model dgcnn --epochs 100 --lr 0.001 --exp_name crosspoint_revise --batch_size 20 --print_freq 200 --k 15

dgcnn.py get_graph_feature也有一个device已改

python train_crosspoint_update.py --model dgcnn --epochs 100 --lr 0.001 --exp_name crosspoint_revise_lam5_warm5 --batch_size 32 --print_freq 200 --k 15 --cudaNo 1
每次需要更改实验名字和使用的cuda编号 batchsize可以支持30的 测试一下num_workers