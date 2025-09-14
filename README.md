# README

## Acknowledgement

This project is built upon the open-source work of [PerFRDiff](https://github.com/xk0720/PerFRDiff). We sincerely thank the authors for their contributions and making their code publicly available. Our work extends and adapts their implementation for further research.

---

## Usage

1. **Dataset and Environment Setup**  
   Please first follow the instructions from [PerFRDiff](https://github.com/xk0720/PerFRDiff) to complete the following configurations:
   - Dataset download and directory organization
   - Python environment setup (including dependencies)
   - Download and placement of required pre-trained weights

2. **Set Pre-trained Model Paths**  
   Before running the code, please modify the following two files to specify the paths to your pre-trained models:

   - **Code A**: `path/to/code_A.py`  
     Update the `video_ckpt_path` variable to point to your **video pre-trained model** checkpoint:

     audio_ckpt_path = "your/path/to/audio_pretrained_model.ckpt"
     ```

3. **Run Training or Inference**  
  The training and inference procedures are the same as in ReactFace. Please refer to the[PerFRDiff](https://github.com/xk0720/PerFRDiff) for commands and usage examples.
