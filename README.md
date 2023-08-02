# CrowdCounting_with_weight

This model requires GPU for training. So please run "check_cuda.py" to check if your computer has GPU.   
Please use "train.py" for training model. I used ShanghaiTech dataset for training.   
"run_libranet.py" is for predict. Change the value of the variable "img_name" to the path to your picture you want to predict.   
Link to dataset:   
https://pan.baidu.com/share/init?surl=VENBbBBbIoS929DMaN5Uug (code: ix2v) - pre-processed ShanghaiTech Part_A training   
https://pan.baidu.com/share/init?surl=lagHgw3gshIBmPTHIbkzRw (code: h7a6) - ShanghaiTech Part_A testing   
https://pan.baidu.com/share/init?surl=V5kVYdyF7Cs5SVlyVm2zGg (code: 3cfp) - VGG16 backbone pretrained on SHT Part_A   
https://1drv.ms/u/s!Atb46kl_Ra3rjkPWiQ5ay3zEwGAG?e=grlZX7 - Pretrained model with the result is mae=59.2706,mse=95.2439 at epoch 32
