wenet 使用介绍



+ **wenet 上手，运行**

  + 安装
    + git clone https://github.com/wenet-e2e/wenet.git
    + 配置conda环境
    + 安装requirements.txt
    + 具体，查看官网
  + 运行
    + 在[examples](https://github.com/wenet-e2e/wenet/tree/main/examples)/[aishell](https://github.com/wenet-e2e/wenet/tree/main/examples/aishell)/**s1**/有案例代码
    + run.sh是主程序
    + 目录分析
      + conf 是特征以及模型配置目录
        + [train_conformer.yaml](https://github.com/wenet-e2e/wenet/blob/main/examples/aishell/s1/conf/train_conformer.yaml) 模型配置文件
          + 主要参数
            + ctc_weight: 0.3
            + batch_size: 20
            + lr: 0.002
            + raw_wav: false
            + 
        + [train_unified_conformer.yaml](https://github.com/wenet-e2e/wenet/blob/main/examples/aishell/s1/conf/train_unified_conformer.yaml) 
          + use_dynamic_chunk: true
      + [local](https://github.com/wenet-e2e/wenet/tree/main/examples/aishell/s1/local) 针对这个corpus 的数据操作脚本
      + tools 常用工具脚本
      + wenet ：软件组织框架目录，底层pytorch, 涉及到模型定义，数据加载，训练主代码，checkpoint等
    + s0 不会保存特征，特征是动态生成的
    + s1会先提取特征
      + 要部署在安卓上，特征不能使用pitch
      + 可以先数据数据成kaldi-format
      + 数据最后通过tools/format_data.sh 生成format.data。这是wenet程序直接读取的数据
    + 在run.sh里面注意项
      + checkpoint=
      + decode_checkpoint
      + decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
      + decoding_chunk_size=

+ **wenet功能**

  + 主要模型CTC + conformer or CTC + transformer

  + 流式/非流式的统一结构。

    + chunk or windows
    + 动态chunk
      + 在训练时，随机为样本选择不同的chunk大小，从而允许一个模型可处理不同chunk大小的输入

  + CTC + attention rescoring ， 将CTC decoder的n-best结果，通过attention decoder进行重打分

  + https://pic1.zhimg.com/v2-342066a83add0645973c07dfa7741d40_b.webp

  + 模型可以量化，Wenet使用dynamic量化来对模型进行压缩

  + 支持了时间戳。解码器不仅可以返回 Nbest 解码结果，而且还可以返回其中每个字对应的时间信息。

  + 支持 Endpoint 检测

  + 支持CTC alignment：输入音频和对应文本，结合维特比算法得到一个字级别的对齐结果。（尖峰特性）

  + WeNet 中选择基于 n-gram 的统计语言模型，结合WFST(Weighted Finite State Transducer)框架和传统语音识别解码技术，实现对定制语言模型的支持。在 AIShell-1, AIShell-2 和 LibriSpeech 三个数据集上，WeNet 的 LM 方案均取得相对错误率3%～10%的下降。

  + 我

  + ![img](https://pic1.zhimg.com/80/v2-da7e038b87784769f1bcb625371abe24_720w.jpg)

  + 利用 Attention Decoder对 CTC 产生的多个候选结果 N-best 进行重打分。

    + 无LM，即依靠CTC prefix beam search生成N-best
    + 有LM，则依靠CTC WFST search生成N-best，WFST search为依靠传统解码图的传统解码方式。
    + 解码图的构建即将建模单元T、词典L、语言模型G各层次信息组合在一张解码图TLG中
    + 解码器则和传统语音识别中解码器一致，使用标准的Viterbi beam search算法解码

  + 基于websocket的业务接入服务

  + wenet1.0发布的更新

    + U2++ 双向建模，
    + ![image-20211009212956846](C:\Users\baba\AppData\Roaming\Typora\typora-user-images\image-20211009212956846.png)
    + 模型层面支持动态left chunk训练

    

+ 代码框架介绍
  + 

