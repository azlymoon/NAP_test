# docker build -t naturalistic-adversarial-patch .

# docker run --gpus all --shm-size=2g -v "//c/Users/danil/Desktop/Naturalistic-Adversarial-Patch/dataset:/usr/src/app/dataset" -v "//c/Users/danil/Desktop/Naturalistic-Adversarial-Patch/exp:/usr/src/app/exp" -it naturalistic-adversarial-patch python ensemble.py --seed 12345 --model yolov5 --classBiggan 259 --epochs 2

# exp6
docker run --gpus all --shm-size=4g -v "/home/tfg-dlg/tfg/NaturalisticAdversarialPatchYOLOv8/dataset:/usr/src/app/dataset" -v "/home/tfg-dlg/tfg/NaturalisticAdversarialPatchYOLOv8/exp:/usr/src/app/exp" -it naturalistic-adversarial-patch python ensemble.py --seed 12345 --model yolov5 --classBiggan 84 --epochs 1000 --weight_loss_tv 0.0

# exp7
docker run --gpus all --shm-size=4g -v "/home/tfg-dlg/tfg/NaturalisticAdversarialPatchYOLOv8/dataset:/usr/src/app/dataset" -v "/home/tfg-dlg/tfg/NaturalisticAdversarialPatchYOLOv8/exp:/usr/src/app/exp" -it naturalistic-adversarial-patch python ensemble.py --seed 12345 --model yolov5 --classBiggan 84 --epochs 1000 --weight_loss_tv 0.1

# exp8
docker run --gpus all --shm-size=4g -v "/home/tfg-dlg/tfg/NaturalisticAdversarialPatchYOLOv8/dataset:/usr/src/app/dataset" -v "/home/tfg-dlg/tfg/NaturalisticAdversarialPatchYOLOv8/exp:/usr/src/app/exp" -it naturalistic-adversarial-patch python ensemble.py --seed 12345 --model yolov8 --classBiggan 84 --epochs 1000 --weight_loss_tv 0.0

# exp9
docker run --gpus all --shm-size=4g -v "/home/tfg-dlg/tfg/NaturalisticAdversarialPatchYOLOv8/dataset:/usr/src/app/dataset" -v "/home/tfg-dlg/tfg/NaturalisticAdversarialPatchYOLOv8/exp:/usr/src/app/exp" -it naturalistic-adversarial-patch python ensemble.py --seed 12345 --model yolov8 --classBiggan 84 --epochs 1000 --weight_loss_tv 0.1