# FoReS (Foundation Recommendation System)

Note: Use the following folder structure to run the code.

Folder Structure:

- /
  - datasets
    - fitzpatrick
      - fairface
        - train
        - val
      - test.csv
      - train.csv
    - foundation
      - maybelline.csv
  - model_checkpoints
    - sam_vit_h_4b8939.pth
    - sam_vit_l_0b3195.pth
  - .gitattributes
  - .gitignore
  - README.md
  - requirements.txt
  - main.ipynb

## Requirements

- Python 3.11.0
- Model Checkpoints from this [Github Repo](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
- Fritzpatrick Dataset from this [Kaggle Link](https://www.kaggle.com/datasets/vinitasilaparasetty/fitzpatrick-classification-by-ethnicity)
