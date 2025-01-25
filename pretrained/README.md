## Example SSL Models

This example uses **wav2vec** models from Fairseq. For more details, visit the [Fairseq wav2vec documentation](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec).

### Pre-trained Model Links
- **`w2v-base`**: [wav2vec_small.pt](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt)  
  A smaller version of the wav2vec model.
  
- **`w2v-large`**: [w2v_large_lv_fsh_swbd_cv.pt](https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv.pt)  
  A large model pre-trained on multiple datasets (LibriVox, Fisher, Switchboard, and CommonVoice).

- **`w2v-xlsr`**: [xlsr2_300m.pt](https://huggingface.co/facebook/wav2vec2-xls-r-300m)  
  A multilingual model with cross-lingual speech representation (XLSR) pre-trained on 300M parameters.