import os
import sys

print(sys.path)
sys.path.append('/home/liyihan/anaconda3')

lr = "0.01"
batch_size = "30"
arch = "resnet18"
data_name = "ucf101"
representation = "mv"
data_root = "data/datalists/ucf101_split1_train.txt"
train_list = " data/datalists/ucf101_split1_train.txt"
test_list = "data/datalists/ucf101_split1_test.txt"
model_prefix = "ucf101_mv_model"
lr_steps = "150 270 390"
epochs = "510"
gpus = "0 "
# no_accumulation = "--no-accumulation"

output_file = "training_"+data_name+"_"+representation+"_bs"+batch_size+"_accumulated"+".out"



# os.system("nohup sh -c '" +
#           sys.executable + " train.py --lr 0.01 >res1.txt && " +
#           sys.executable + " train.py --lr 0.03 >res2.txt && " +
#           sys.executable + " train.py --lr 0.09 >res3.txt" +
#           "' &")
# print("Enable anaconda")

# if(os.system("conda activate coviar")==0):
    # print(os.getcwd())
print("Create output file")
if os.path.exists(output_file):
    print("Exists.")
else:
    os.system("touch "+ output_file)
    if os.access(output_file, os.W_OK):
        print("Start executing")
        os.system("conda activate coviar")
        print(os.system("nohup sh -c '" +
                sys.executable + " train.py --lr " + lr + " --batch-size " + batch_size + " --arch " + arch +
                " --data-name " + data_name + " --representation " + representation +
                " --data-root " + data_root + " --train-list " + train_list + " --test-list " + test_list +
                " --model-prefix " + model_prefix + " --lr-steps " + lr_steps + " --epochs " + epochs +
                " --gpus " + gpus +
                " > " + output_file +
                "' &"))

print("End")




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