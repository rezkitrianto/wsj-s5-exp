#!/bin/bash


# run_tdnn_lstm_1a.sh is a TDNN+LSTM system.  Compare with the TDNN
# system in run_tdnn_1a.sh.  Configuration is similar to
# the same-named script run_tdnn_lstm_1a.sh in
# egs/tedlium/s5_r2/local/nnet3/tuning.

# It's a little better than the TDNN-only script on dev93, a little
# worse on eval92.

# steps/info/nnet3_dir_info.pl exp/nnet3/tdnn_lstm1a_sp
# exp/nnet3/tdnn_lstm1a_sp: num-iters=102 nj=3..10 num-params=8.8M dim=40+100->3413 combine=-0.55->-0.54 loglike:train/valid[67,101,combined]=(-0.63,-0.55,-0.55/-0.71,-0.63,-0.63) accuracy:train/valid[67,101,combined]=(0.80,0.82,0.82/0.76,0.78,0.78)



# local/nnet3/compare_wer.sh --looped --online exp/nnet3/tdnn1a_sp exp/nnet3/tdnn_lstm1a_sp 2>/dev/null
# local/nnet3/compare_wer.sh --looped --online exp/nnet3/tdnn1a_sp exp/nnet3/tdnn_lstm1a_sp
# System                tdnn1a_sp tdnn_lstm1a_sp
#WER dev93 (tgpr)                9.18      8.54
#             [looped:]                    8.54
#             [online:]                    8.57
#WER dev93 (tg)                  8.59      8.25
#             [looped:]                    8.21
#             [online:]                    8.34
#WER dev93 (big-dict,tgpr)       6.45      6.24
#             [looped:]                    6.28
#             [online:]                    6.40
#WER dev93 (big-dict,fg)         5.83      5.70
#             [looped:]                    5.70
#             [online:]                    5.77
#WER eval92 (tgpr)               6.15      6.52
#             [looped:]                    6.45
#             [online:]                    6.56
#WER eval92 (tg)                 5.55      6.13
#             [looped:]                    6.08
#             [online:]                    6.24
#WER eval92 (big-dict,tgpr)      3.58      3.88
#             [looped:]                    3.93
#             [online:]                    3.88
#WER eval92 (big-dict,fg)        2.98      3.38
#             [looped:]                    3.47
#             [online:]                    3.53
# Final train prob        -0.7200   -0.5492
# Final valid prob        -0.8834   -0.6343
# Final train acc          0.7762    0.8154
# Final valid acc          0.7301    0.7849


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
srcpath=/home/rezki/sandbox/kaldi/egs/chime4DevFE1_allChannels3/s5_6ch/
destpath=/home/rezki/sandbox/kaldi/egs/chime4DevFE1_allChannels3/s5_6ch/

stage=14
nj=30
train_set=tr05_multi_noisy
test_sets="dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics"
test_sets="et05_simu_beamformit_5mics dt05_simu_beamformit_5mics dt05_real_beamformit_5mics"
test_sets="et05_real_beamformit_5mics"
gmm=tri4b        # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1a  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM options
train_stage=-10
label_delay=5

# training chunk-options
chunk_width=40,30,20
chunk_left_context=40
chunk_right_context=0
xent_regularize=0.025

# training options
srand=0
remove_egs=true

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.

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

# /home/rezki/sandbox/kaldi/egs/chime4DevFE1_allChannels3/s5_6ch/local/nnet3/run_ivector_common.sh || exit 1;
# ${srcpath}local/nnet3/run_ivector_common.sh || exit 1;

# exit 1

# local/nnet3/run_ivector_common.sh \
#   --stage $stage --nj $nj \
#   --train-set $train_set --gmm $gmm \
#   --num-threads-ubm $num_threads_ubm \
#   --nnet3-affix "$nnet3_affix"

gmm_res=${srcpath}exp/tri3b_tr05_multi_noisy
gmm_ali=${srcpath}exp/tri3b_tr05_multi_noisy_ali
data_fmllr=${srcpath}data-fmllr-tri3b

gmm_dir=$gmm_res
ali_dir=$gmm_ali
lat_dir=${srcpath}exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats

dir=${destpath}exp/chain${nnet3_affix}/tdnn3${affix}_sp_blstm #_v3_5layers

train_data_dir=$data_fmllr/tr05_multi_noisy
train_ivector_dir=${srcpath}exp/nnet3/ivectors_tr05_multi_noisy_fmllr
lores_train_data_dir=${data_fmllr}/${train_set}

# gmm_dir=exp/${gmm}
# ali_dir=exp/${gmm}_ali_${train_set}_sp
# lang=data/lang
# dir=exp/nnet3${nnet3_affix}/tdnn_lstm${affix}_sp
# train_data_dir=data/${train_set}_sp_hires
# train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $gmm_dir/graph_tgpr_5k/HCLG.fst \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 12 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $ali_dir/tree |grep num-pdfs|awk '{print $2}')
  [ -z $num_targets ] && { echo "$0: error getting num-targets"; exit 1; }
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  lstm_opts="decay-time=20"
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat delay=$label_delay

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

  fast-lstmp-layer name=blstm4-forward input=Append(blstm3-forward, blstm3-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm4-backward input=Append(blstm3-forward, blstm3-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 $lstm_opts

  fast-lstmp-layer name=blstm5-forward input=Append(blstm4-forward, blstm4-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm5-backward input=Append(blstm4-forward, blstm4-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 $lstm_opts


  ## adding the layers for chain branch
  output-layer name=output input=Append(blstm5-forward, blstm5-backward) output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=Append(blstm5-forward, blstm5-backward) output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$dir/egs/storage $dir/egs/storage
  fi
  # train_stage=277
  steps/nnet3/train_rnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=5 \
    --trainer.deriv-truncate-margin=10 \
    --trainer.samples-per-iter=20000 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.0003 \
    --trainer.optimization.final-effective-lrate=0.00003 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=64,32 \
    --trainer.optimization.momentum=0.5 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --feat-dir=$train_data_dir \
    --ali-dir=$ali_dir \
    --lang=$lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi

# exit 1

if [ $stage -le 14 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
      data_affix=$(echo $data | sed s/test_//)
      nj=$(wc -l <${data_fmllr}/${data}/spk2utt)
      for lmtype in graph_tgpr_5k; do
        graph_dir=$gmm_dir/graph_${lmtype}
        graph_dir2=/home/rezki/sandbox/kaldi/egs/chime4DevFE1_allChannels3/s5_6ch/exp/tri4a_dnn_tr05_multi_noisy/graph_tgpr_5k
        steps/nnet3/decode.sh \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nj --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir ${srcpath}exp/nnet3${nnet3_affix}/ivectors_${data}_fmllr \
          $graph_dir2 ${data_fmllr}/${data} ${dir}/decode_${lmtype}_${data_affix} || exit 1
      done

      # steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
      #   data/${data}_hires ${dir}/decode_{tgpr,tg}_${data_affix} || exit 1
      # steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      #   data/lang_test_bd_{tgpr,fgconst} \
      #  data/${data}_hires ${dir}/decode_${lmtype}_${data_affix}{,_fg} || exit 1

    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 1

if [ $stage -le 15 ]; then
  # 'looped' decoding.
  # note: you should NOT do this decoding step for setups that have bidirectional
  # recurrence, like BLSTMs-- it doesn't make sense and will give bad results.
  # we didn't write a -parallel version of this program yet,
  # so it will take a bit longer as the --num-threads option is not supported.
  # we just hardcode the --frames-per-chunk option as it doesn't have to
  # match any value used in training, and it won't affect the results (unlike
  # regular decoding).
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nj=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in graph_tgpr_5k; do
        graph_dir=$gmm_dir/graph_${lmtype}
        steps/nnet3/decode_looped.sh \
          --frames-per-chunk 30 \
          --nj $nj --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $graph_dir data/${data}_hires ${dir}/decode_looped_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
        data/${data}_hires ${dir}/decode_looped_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}/decode_looped_${lmtype}_${data_affix}{,_fg} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if $test_online_decoding && [ $stage -le 16 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nj=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      for lmtype in tgpr bd_tgpr; do
        graph_dir=$gmm_dir/graph_${lmtype}
        steps/online/nnet3/decode.sh \
          --nj $nj --cmd "$decode_cmd" \
          $graph_dir data/${data} ${dir}_online/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
        data/${data}_hires ${dir}_online/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}_online/decode_${lmtype}_${data_affix}{,_fg} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi



exit 0;
