#!/bin/bash
echo "*******************1************************"
echo "Evaluating 1/ at beam_size = 1"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 1 -ch /home/enes/mmi727_project/trainings/1/BEST_checkpoint_10_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/1/config.yaml \
|& tee /home/enes/mmi727_project/trainings/1/eval_beam_1.txt

echo
echo "Evaluating 1/ at beam_size = 5"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 5 -ch /home/enes/mmi727_project/trainings/1/BEST_checkpoint_10_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/1/config.yaml \
|& tee /home/enes/mmi727_project/trainings/1/eval_beam_5.txt


echo
echo
echo "*******************2************************"
echo "Evaluating 2/ at beam_size = 1"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 1 -ch /home/enes/mmi727_project/trainings/2/BEST_checkpoint_13_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/2/config.yaml \
|& tee /home/enes/mmi727_project/trainings/2/eval_beam_1.txt

echo
echo "Evaluating 2/ at beam_size = 5"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 5 -ch /home/enes/mmi727_project/trainings/2/BEST_checkpoint_13_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/2/config.yaml \
|& tee /home/enes/mmi727_project/trainings/2/eval_beam_5.txt


echo
echo
echo "*******************3************************"
echo "Evaluating 3/ at beam_size = 1"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 1 -ch /home/enes/mmi727_project/trainings/3/BEST_checkpoint_7_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/3/config.yaml \
|& tee /home/enes/mmi727_project/trainings/3/eval_beam_1.txt

echo
echo "Evaluating 3/ at beam_size = 5"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 5 -ch /home/enes/mmi727_project/trainings/3/BEST_checkpoint_7_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/3/config.yaml \
|& tee /home/enes/mmi727_project/trainings/3/eval_beam_5.txt



echo
echo
echo "*******************4************************"
echo "Evaluating 4/ at beam_size = 1"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 1 -ch /home/enes/mmi727_project/trainings/4/BEST_checkpoint_14_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/4/config.yaml \
|& tee /home/enes/mmi727_project/trainings/4/eval_beam_1.txt

echo
echo "Evaluating 4/ at beam_size = 5"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 5 -ch /home/enes/mmi727_project/trainings/4/BEST_checkpoint_14_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/4/config.yaml \
|& tee /home/enes/mmi727_project/trainings/4/eval_beam_5.txt


echo
echo
echo "*******************5************************"
echo "Evaluating 5/ at beam_size = 1"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 1 -ch /home/enes/mmi727_project/trainings/5/BEST_checkpoint_11_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/5/config.yaml \
|& tee /home/enes/mmi727_project/trainings/5/eval_beam_1.txt

echo
echo "Evaluating 5/ at beam_size = 5"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 5 -ch /home/enes/mmi727_project/trainings/5/BEST_checkpoint_11_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/5/config.yaml \
|& tee /home/enes/mmi727_project/trainings/5/eval_beam_5.txt


echo
echo
echo "*******************6************************"
echo "Evaluating 6/ at beam_size = 1"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 1 -ch /home/enes/mmi727_project/trainings/6/BEST_checkpoint_8_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/6/config.yaml \
|& tee /home/enes/mmi727_project/trainings/6/eval_beam_1.txt

echo
echo "Evaluating 6/ at beam_size = 5"
/home/enes/mmi727_project/myEnvironment/showAttendTell/bin/python3 \
eval.py -b 5 -ch /home/enes/mmi727_project/trainings/6/BEST_checkpoint_8_coco_5_cap_per_img_5_min_word_freq.pth.tar \
-cfg /home/enes/mmi727_project/trainings/6/config.yaml \
|& tee /home/enes/mmi727_project/trainings/6/eval_beam_5.txt