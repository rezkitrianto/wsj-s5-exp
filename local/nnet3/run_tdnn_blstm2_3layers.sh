#!/bin/bash

# tdnn_blstm_1a is same as blstm_6k, but with the initial tdnn layers
# blstm_6k : num-parameters: 41155430
# tdnn_blstm_1a : num-parameters: 53688166

# local/chain/compare_wer_general.sh blstm_6l_sp blstm_6k_sp
# System                blstm_6k_sp tdnn_blstm_1a_sp
# WER on train_dev(tg)      13.25     12.95
# WER on train_dev(fg)      12.27     11.98
# WER on eval2000(tg)        15.7      15.5
# WER on eval2000(fg)        14.5      14.1
# Final train prob         -0.052    -0.041
# Final valid prob         -0.080    -0.072
# Final train prob (xent)        -0.743    -0.629
# Final valid prob (xent)       -0.8816   -0.8091

set -e

srcpath=/home/rezki/sandbox/kaldi/egs/chime4DevFE1_allChannels3/s5_6ch/
destpath=/home/rezki/sandbox/kaldi/egs/chime4DevFE1_allChannels3/s5_6ch/
# srcpath=/mnt/Data/xlib/kaldi/egs/chime4DevFE1_allChannels3/s5_6ch/
# destpath=/media/alluser/rezki/kaldi/
# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
# dir=${srcpath}exp/chain/tdnn_blstm_1a_5layers_dropout1_noIvector_real  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_dir_affix=

# training options
leftmost_questions_truncate=-1
chunk_width=150
chunk_left_context=40
chunk_right_context=40
xent_regularize=0.025
self_repair_scale=0.00001
label_delay=0
dropout_schedule='0,0@0.20,0.1@0.50,0'

# decode options
extra_left_context=50
extra_right_context=50
frames_per_chunk=

remove_egs=false
common_egs_dir=

affix=
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=$dir${affix:+_$affix}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
dir=${dir}$suffix

dir=${srcpath}exp/chain/tdnn_blstm_1a_3layers_dropout1_sp_noIvector
# dir=/home/rezki/sandbox/kaldi/egs/chime4DevFE1_allChannels3/s5_6ch/exp/chain/tdnn_blstm_1a_5layers_sp
train_set=tr05_multi_noisy
test_sets="dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics"
# test_sets="dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_simu_beamformit_5mics"
# test_sets="et05_real_beamformit_5mics"
gmm_res=${srcpath}exp/tri3b_tr05_multi_noisy
gmm_ali=${srcpath}exp/tri3b_tr05_multi_noisy_ali
data_fmllr=${srcpath}data-fmllr-tri3b

gmm_dir=$gmm_res
ali_dir=$gmm_ali
lat_dir=${srcpath}exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
lat_dir=${srcpath}exp/tri3b_tr05_multi_noisy_ali
lang=${srcpath}data/lang_chain_2y
# ali_dir=exp/tri4_ali_nodup$suffix
treedir=${srcpath}exp/chain/tri5_7d_tree$suffix
# lang=data/lang_chain_2y

train_data_dir=$data_fmllr/tr05_multi_noisy
train_ivector_dir=${srcpath}exp/nnet3/ivectors_tr05_multi_noisy_fmllr

# if we are using the speed-perturbed data we need to generate
# alignments for it.
# local/nnet3/run_ivector_common.sh --stage $stage \
#   --speed-perturb $speed_perturb \
#   --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat ${srcpath}exp/tri3b_tr05_multi_noisy_ali/num_jobs) || exit 1;
  # steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
  #   data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${srcpath}data/$train_set \
    ${srcpath}data/lang ${srcpath}exp/tri3b_tr05_multi_noisy ${srcpath}exp/tri3b_tr05_multi_noisy_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
fi

lat_dir2=${srcpath}exp/tri3b_tr05_multi_noisy_lats_nodup$suffix

# exit 1

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r ${srcpath}data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 $train_data_dir $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  [ -z $num_targets ] && { echo "$0: error getting num-targets"; exit 1; }
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  lstm_opts="decay-time=20 dropout-proportion=0.0"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  #input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  #fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=1024
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=1024

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=blstm1-forward input=tdnn3 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm1-backward input=tdnn3 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 $lstm_opts

  fast-lstmp-layer name=blstm2-forward input=Append(blstm1-forward, blstm1-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm2-backward input=Append(blstm1-forward, blstm1-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 $lstm_opts

  fast-lstmp-layer name=blstm3-forward input=Append(blstm2-forward, blstm2-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm3-backward input=Append(blstm2-forward, blstm2-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=Append(blstm3-forward, blstm3-backward) output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=Append(blstm3-forward, blstm3-backward) output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi
# --feat.online-ivector-dir $train_ivector_dir \
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 32 \
    --trainer.frames-per-iter 1200000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 4 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 1 \
    --trainer.optimization.num-jobs-final 1 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --trainer.dropout-schedule $dropout_schedule \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir2 \
    --dir $dir  || exit 1;
fi

# exit 1

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${srcpath}data/lang_test_tgpr_5k $dir $dir/graph_sw1_tg
fi

# exit 1

has_fisher=false
decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
graph_dir2=/home/rezki/sandbox/kaldi/egs/chime4DevFE1_allChannels3/s5_6ch/exp/tri4a_dnn_tr05_multi_noisy/graph_tgpr_5k
graph_dir_name=graph_tgpr_5k
if [ $stage -le 15 ]; then
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=$chunk_width;
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi

  # steps/online/nnet3/prepare_online_decoding.sh --mfcc-config ${srcpath}conf/mfcc_hires.conf ${srcpath}data/lang ${srcpath}exp/nnet3/extractor_fmllr $dir ${dir}_online
  # exit 1
  for decode_set in $test_sets; do
      (
      # --online-ivector-dir ${srcpath}exp/nnet3${nnet3_affix}/ivectors_${decode_set}_fmllr \
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --cmd "$decode_cmd" $iter_opts \
          --extra-left-context $extra_left_context  \
          --extra-right-context $extra_right_context  \
          --frames-per-chunk "$frames_per_chunk" \
         $graph_dir ${data_fmllr}/${decode_set} \
         $dir/decode2_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_${decode_suff} || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            ${srcpath}data/lang ${srcpath}data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_${graph_dir_name} || exit 1;
      fi
      ) &
  done
fi

exit 1

wait;
exit 0;
