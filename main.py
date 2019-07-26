import os
import sys

lr = "0.01"
batch_size = "30"
arch = "resnet18"
data_name = "ucf101"
representation = "mv"
data_root = "data/ucf101/mpeg4_videos"
train_list = " data/datalists/ucf101_split1_train.txt"
test_list = "data/datalists/ucf101_split1_test.txt"
model_prefix = "ucf101_mv_model"
lr_steps = "150 270 390"
epochs = "510"
gpus = "0 "
is_accumulated = True

accu_option = ""
if(is_accumulated):
    accu_option = ""
else:
    accu_option = " --no_accumulation "

if is_accumulated:
    output_file = "training_"+data_name+"_"+representation+"_bs"+batch_size+"_accumulated"+".out"
    # model_checkpoint = model_prefix+"_"+representation+"_checkpoint"+"_bs"+batch_size+"_accumulated"+".pth.tar"
    # model_best = model_prefix+"_"+representation+"_best"+"_bs"+batch_size+"_accumulated"+".pth.tar"
else:
    output_file = "training_" + data_name + "_" + representation + "_bs" + batch_size + "_original" + ".out"
    # model_checkpoint = model_prefix + "_" + representation + "_checkpoint" + "_bs" + batch_size + "_original" + ".pth.tar"
    # model_best = model_prefix + "_" + representation + "_best" + "_bs" + batch_size + "_original" + ".pth.tar"

# print(os.getcwd())
# print(sys.executable)

print("Create output file")
if os.path.exists(output_file):
    print("File exists")
else:
    os.system("touch "+ output_file)
    if os.access(output_file, os.W_OK):
        print("Enable coviar")
        os.system("/home/liyihan/anaconda3/bin/activate coviar")
        print("Start training")
        os.system("nohup sh -c '" +
                sys.executable + " train.py --lr " + lr + " --batch-size " + batch_size + " --arch " + arch +
                " --data-name " + data_name + " --representation " + representation +
                " --data-root " + data_root + " --train-list " + train_list + " --test-list " + test_list +
                " --model-prefix " + model_prefix + " --lr-steps " + lr_steps + " --epochs " + epochs +
                " --gpus " + gpus + accu_option +
                " > " + output_file +
                "' &")



'''
nohup python train.py --lr 0.01 --batch-size 30 --arch resnet18 \
 	--data-name ucf101 --representation mv \
 	--data-root data/ucf101/mpeg4_videos \
 	--train-list data/datalists/ucf101_split1_train.txt \
 	--test-list data/datalists/ucf101_split1_test.txt \
 	--model-prefix ucf101_mv_model \
 	--lr-steps 150 270 390  --epochs 510 \
 	--gpus 0 &
'''

'''
python train.py --lr 0.01 --batch-size 30 --arch resnet18 \
 	--data-name ucf101 --representation mv \
 	--data-root data/ucf101/mpeg4_videos \
 	--train-list data/datalists/ucf101_split1_train.txt \
 	--test-list data/datalists/ucf101_split1_test.txt \
 	--model-prefix ucf101_mv_model \
 	--lr-steps 150 270 390  --epochs 510 \
 	--gpus 0 &

python train.py --lr 0.01 --batch-size 40 --arch resnet18 \
 	--data-name ucf101 --representation mv \
 	--data-root data/ucf101/mpeg4_videos \
 	--train-list data/datalists/ucf101_split1_train.txt \
 	--test-list data/datalists/ucf101_split1_test.txt \
 	--model-prefix ucf101_mv_model \
 	--lr-steps 150 270 390  --epochs 510 \
 	--gpus 0 &
'''

'''
# I-frame model.
python train.py --lr 0.0003 --batch-size 1 --arch resnet152 \
 	--data-name ucf101 --representation iframe \
 	--data-root data/ucf101/mpeg4_videos \
 	--train-list data/datalists/ucf101_split1_train.txt \
 	--test-list data/datalists/ucf101_split1_test.txt \
 	--model-prefix ucf101_iframe_model \
 	--lr-steps 150 270 390  --epochs 510 \
 	--gpus 0

'''

