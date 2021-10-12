#!/bin/bash

declare -A execution_times

case "$2" in
	CIFAR-10)
		execution_times[Random]=04:00:00
		execution_times[Random_SteppedLR]=04:00:00
		execution_times[Lorraine]=05:30:00
		execution_times[Baydin]=05:30:00
		execution_times[Ours_LR]=05:30:00
		execution_times[Ours_LR_Momentum]=05:30:00
		execution_times[Ours_HDLR_Momentum]=12:00:00
		execution_times[DiffThroughOpt]=05:30:00
		mkdir -p "/rds/user/rmc78/hpc-work/ShortHorizonBias/ICLR_CIFAR10/$1"
		output_path="/rds/user/rmc78/hpc-work/ShortHorizonBias/ICLR_CIFAR10/$1/%a_stdout.txt"
		error_path="/rds/user/rmc78/hpc-work/ShortHorizonBias/ICLR_CIFAR10/$1/%a_stderr.txt"
		write_path="/rds/user/rmc78/hpc-work/ShortHorizonBias/ICLR_CIFAR10/"
		;;
	PennTreebank)
		execution_times[Random]=06:00:00
		execution_times[Random_Validation]=07:30:00
		execution_times[Random_SteppedLR]=06:30:00
		execution_times[Lorraine]=07:30:00
		execution_times[Baydin]=08:00:00
		execution_times[Ours_LR]=08:00:00
		execution_times[Ours_LR_Momentum]=08:00:00
		#execution_times[Ours_HDLR_Momentum]=17:30:00
		execution_times[Ours_HDLR_Momentum]=12:00:00
		execution_times[DiffThroughOpt]=08:00:00
		mkdir -p "/rds/user/rmc78/hpc-work/ShortHorizonBias/ICLR_PennTreebank/$1"
		output_path="/rds/user/rmc78/hpc-work/ShortHorizonBias/ICLR_PennTreebank/$1/%a_stdout.txt"
		error_path="/rds/user/rmc78/hpc-work/ShortHorizonBias/ICLR_PennTreebank/$1/%a_stderr.txt"
		write_path="/rds/user/rmc78/hpc-work/ShortHorizonBias/ICLR_PennTreebank/"
		;;
esac

subconfig=$1 write_path=$write_path \
	sbatch \
	--time=${execution_times[$1]} \
	--output=${output_path} \
	--error=${error_path} \
	$2/$2_default.slurm
