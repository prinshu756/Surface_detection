@echo off
cd /d D:\MachineLearning\industrial_surface_defect
"C:\Users\Prinshu\AppData\Local\Programs\Python\Python312\pythonw.exe" src\train.py --arch segformer --encoder mit_b2 --batch 10 --epochs 10 --lr 2e-4 --pw_factor 0.15 --oversample "defect_a:3,defect_b:8,defect_d:2" --init_from checkpoints\segformer_mit_b2_best.pt --tag segformer_mit_b2_os --resume > outputs\train_os.out.log 2> outputs\train_os.err.log
