先各自训练gmm tri3
然后生成对齐。

在 train_sat时候会生成对齐。根据对齐
统一tree
再次训练

合并对齐

combine data

combine alignment

steps/combine_ali_dirs.sh --nj 32 data/train_aug/mfcc-pitch/ exp/tri4_ali_combined ../id_appen/exp/tri4a/ali_train/ ../id_youtube/exp/tri4a/ali_train/ ../id_huang/exp/tri4a/ali_train/






#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# This does Speaker Adapted Training (SAT), i.e. train on
# fMLLR-adapted features.  It can be done on top of either LDA+MLLT, or
# delta and delta-delta features.  If there are no transforms supplied
# in the alignment directory, it will estimate transforms itself before
# building the tree (and in any case, it estimates transforms a number
# of times during training).


# Begin configuration section.
stage=-5
exit_stage=-100 # you can use this to require it to exit at the
                # beginning of a specific stage.  Not all values are
                # supported.
fmllr_update_type=full
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
careful=false
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
context_opts=  # e.g. set this to "--context-width 5 --central-position 2" for quinphone.
realign_iters="10 20 30";
fmllr_iters="2 4 6 12";
silence_weight=0.0 # Weight on silence in fMLLR estimation.
num_iters=35   # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
power=0.2 # Exponent for number of gaussians according to occurrence counts
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
phone_map=
train_tree=true
tree_stats_opts=
cluster_phones_opts=
compile_questions_opts=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: steps/train_sat.sh <#leaves> <#gauss> <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_sat.sh 2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6

for f in $data/feats.scp $lang/phones.txt $alidir/final.mdl $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "train_sat.sh: no such file $f" && exit 1;
done

numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$max_iter_inc]  # per-iter #gauss increment
oov=`cat $lang/oov.int`
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
sdata=$data/split$nj;
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
delta_opts=`cat $alidir/delta_opts 2>/dev/null`
phone_map_opt=
[ ! -z "$phone_map" ] && phone_map_opt="--phone-map='$phone_map'"

mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null # frame-splicing options.
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
cp $alidir/delta_opts $dir 2>/dev/null # delta option.

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

echo $nj >$dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

# Set up features.

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

## Set up speaker-independent features.
case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir
    cp $alidir/full.mat $dir 2>/dev/null
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

## Get initial fMLLR transforms (possibly from alignment dir)
if [ -f $alidir/trans.1 ]; then
  echo "$0: Using transforms from $alidir"
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
  cur_trans_dir=$alidir
else
  if [ $stage -le -5 ]; then
    echo "$0: obtaining initial fMLLR transforms since not present in $alidir"
    # The next line is necessary because of $silphonelist otherwise being incorrect; would require
    # old $lang dir which would require another option.  Not needed anyway.
    [ ! -z "$phone_map" ] && \
       echo "$0: error: you must provide transforms if you use the --phone-map option." && exit 1;
    $cmd JOB=1:$nj $dir/log/fmllr.0.JOB.log \
      ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post $silence_weight $silphonelist $alidir/final.mdl ark:- ark:- \| \
      gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
      --spk2utt=ark:$sdata/JOB/spk2utt $alidir/final.mdl "$sifeats" \
      ark:- ark:$dir/trans.JOB || exit 1;
  fi
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"
  cur_trans_dir=$dir
fi


if [ $stage -le -2 ]; then
echo "copy tree from exit tree"
 echo "那我走？"
     cp ../my-ntu/tri4a/tree $dir/ || exit 1;
         $cmd JOB=1 $dir/log/init_model.log \
		       gmm-init-model-flat $dir/tree $lang/topo $dir/1.mdl \
		               "$feats subset-feats ark:- ark:-|" || exit 1;
fi

if [ $stage -le -1 ]; then
  # Convert the alignments.
  echo "$0: Converting alignments from $alidir to use current tree"
  $cmd JOB=1:$nj $dir/log/convert.JOB.log \
    convert-ali $phone_map_opt $alidir/final.mdl $dir/1.mdl $dir/tree \
     "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

[ "$exit_stage" -eq 0 ] && echo "$0: Exiting early: --exit-stage $exit_stage" && exit 0;

if [ $stage -le 0 ] && [ "$realign_iters" != "" ]; then
  echo "$0: Compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/1.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

x=1
while [ $x -lt $num_iters ]; do
   echo Pass $x
  if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
    echo Aligning data
    mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
    $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
      gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
      "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
      "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
  fi

  if echo $fmllr_iters | grep -w $x >/dev/null; then
    if [ $stage -le $x ]; then
      echo Estimating fMLLR transforms
      # We estimate a transform that's additional to the previous transform;
      # we'll compose them.
      $cmd JOB=1:$nj $dir/log/fmllr.$x.JOB.log \
        ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
        weight-silence-post $silence_weight $silphonelist $dir/$x.mdl ark:- ark:- \| \
        gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
        --spk2utt=ark:$sdata/JOB/spk2utt $dir/$x.mdl \
        "$feats" ark:- ark:$dir/tmp_trans.JOB || exit 1;
      for n in `seq $nj`; do
        ! ( compose-transforms --b-is-affine=true \
          ark:$dir/tmp_trans.$n ark:$cur_trans_dir/trans.$n ark:$dir/composed_trans.$n \
          && mv $dir/composed_trans.$n $dir/trans.$n && \
          rm $dir/tmp_trans.$n ) 2>$dir/log/compose_transforms.$x.log \
          && echo "$0: Error composing transforms" && exit 1;
      done
    fi
    feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |"
    cur_trans_dir=$dir
  fi

  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;
    [ `ls $dir/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-est --power=$power --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc
    rm $dir/$x.occs
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
done


if [ $stage -le $x ]; then
  # Accumulate stats for "alignment model"-- this model is
  # computed with the speaker-independent features, but matches Gaussian-for-Gaussian
  # with the final speaker-adapted model.
  $cmd JOB=1:$nj $dir/log/acc_alimdl.JOB.log \
    ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$sifeats" \
    ark,s,cs:- $dir/$x.JOB.acc || exit 1;
  [ `ls $dir/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
  # Update model.
  $cmd $dir/log/est_alimdl.log \
    gmm-est --power=$power --remove-low-count-gaussians=false $dir/$x.mdl \
    "gmm-sum-accs - $dir/$x.*.acc|" $dir/$x.alimdl  || exit 1;
  rm $dir/$x.*.acc
fi

rm $dir/final.{mdl,alimdl,occs} 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs
ln -s $x.alimdl $dir/final.alimdl


steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir

utils/summarize_warnings.pl $dir/log
(
  echo "$0: Likelihood evolution:"
  for x in `seq $[$num_iters-1]`; do
    tail -n 30 $dir/log/acc.$x.*.log | awk '/Overall avg like/{l += $(NF-3)*$(NF-1); t += $(NF-1); }
        /Overall average logdet/{d += $(NF-3)*$(NF-1); t2 += $(NF-1);}
        END{ d /= t2; l /= t; printf("%s ", d+l); } '
  done
  echo
) | tee $dir/log/summary.log


steps/info/gmm_dir_info.pl $dir

echo "$0: done training SAT system in $dir"

exit 0
