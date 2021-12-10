# Reproducibility Challenge

## 1. Brief:
SYDE 671 Course Project to replicate and validate paper
### 1.1 Reference:
- Paper: [ICLR 2021 - Colorization Transformer](https://openreview.net/forum?id=5NA1PinlGFu)
- Original Author GitHub: https://github.com/google-research/google-research/tree/master/coltran
- Dataset: [ImageNet 2012]()
- Pre-trained Models: [Original Author's Pre-trained Checkpoints](https://console.cloud.google.com/storage/browser/gresearch/coltran)


## 2. Instructions:
- In addition to the original author's instructions from [ColTran GitHub](https://github.com/google-research/google-research/tree/master/coltran)

### 2.1 Evaluating Custom Dataset with Pre-trained Model
1. Download the model from [Original Author's Pre-trained Checkpoints](https://console.cloud.google.com/storage/browser/gresearch/coltran), and move the unzipped 'coltran' pre-trained model folder under `coltran-open-review/coltran/..`
2. Prepare dataset directory, which can be defined in `coltran-open-review/coltran_custom_run.py` in line `:97-:118`
    - A directory of subdirectories of images: modify the code relate to `MASTER_DIRECTORY`.
    - A directory of image: define your own TAG attributes for defining `CONFIG[COLORTRAN_STEPS.INIT]["image_directory"] = ['coltran/potatoes_images/imgs']` as shown in the code.
3. Run the script: `$ python coltran_custom_run.py {TAG} {OPTIONAL:RUN_STEPS}`
    - It will generates the result in the following mahierarchy:
        ```
        - coltran-open-review
            - coltran
                - result-{TAG}
                   |- imagenet
                        |- color ........................ 256x256x3 colored images
                            |- ...*.jpg
                        |- color_64 ..................... 64x64x3 colored images
                            |- ...*.jpg
                        |- gray
                            |- ...*.jpg
                        |- gray_64
                            |- ...*.jpg
                   |- stage_1
                        |- ...*.jpg ..................... 64x64x3 colored images
                   |- stage_2
                        |- ...*.jpg ..................... 64x64x3 colored images
                   |- stage_3 
                        |- ...*.jpg ..................... 256x256x3 colored images
        ```
    - To run through all steps, leave `{OPTIONAL:RUN_STEPS}` empty
    - To run specific steps, mention all below keywords within `{OPTIONAL:RUN_STEPS}`
      - "init": initialize pre-compilation of dataset resizing and gray images under `coltran/result-{TAG}/imagenet/*`
      - "colorize": uses the data generated from "init" to colorize the image, and output under `coltran/result-{TAG}/stage_1/*`
      - "color_upsampler": uses the data from "stage_1" to upsample the color to a full range color, and output under `coltran/result-{TAG}/stage_2/*`
      - "spatial_upsampler": uses the data from "stage_2" to upsample the image to a full resolution image, and output under `coltran/result-{TAG}/stage_3/*`
  
### 2.2 Evaluating the generated colorized image from 2.1 above
1. Modify the "[USER-INPUT]" section in line 44-48, where the script will auto-generate a lookup table for the pipeline to automatically going through images generated through the 2.1 scripts. The output can be obtained at the terminal output.
2. Run the program:
   - You may invoke: `$ python coltran_evaluate_fid.py`
   - Or save the output to a file: `python coltran_evaluate_fid.py > output.txt`, similar to [example-output-of-FID-scores](coltran/batch123.txt)
3. See the plot under `output`, similar to 
   ![example-of-the-FID-plot](output/plot_FID%20Score%20At%20Different%20Stage.png)
### 2.3 Training and Ablation Training with Validation:
1. Be sure to pre-generate the training images via 2.1, pre-compile with: `$ python coltran_custom_run.py -train`
2. Modify the directory in CONFIG dictionary in the `coltran_custom_train.py`
3. Run: `$ python coltran_custom_train.py {TAG}`:
   - `{TAG}` = `train`: run training through all three stages 
   - `{TAG}` = `train-10K`: run training for 10,000 steps
   - `{TAG}` = `train-ablation-10K`: run ablation training only on Coltran Core (Stage 1 - Coloriztion) for 10,000 steps, with all combinations, includes: baseline axial transformer, no cLN, no cMLP, no cAtt
   - `{TAG}` = `train-ablation-10K-eval`: run evaluation through all the checkpoints for the training trained previously only with Coltran Core (Stage 1 - Coloriztion)
   - `{TAG}` = `train-ablation-10K-validate`: `train-ablation-10K` + validate the model at once per defined periods to evaluate the trained model at that particular step
4. Note: To continuing previously trained models, please modify the folder name and `MAX_TRAIN` in the script:
   - Modify both `{TAG}` and folder name to `train-ablation-20K` instead of `train-ablation-10K`
   - `MAX_TRAIN = 20000` in `coltran_custom_train.py`
   - The pipeline will automatically pick up the latest model in the log directory, and continuing training as long as you set `MAX_TRAIN` to a greater value

