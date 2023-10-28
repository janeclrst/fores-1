# FoReS (Foundation Recommendation System)

Note: Use the following folder structure to run the code.

Folder Structure:

- fores/
  - datasets/
    - fitzpatrick/
      - fairface/
        - train/
          - ....jpeg
        - val/
          - ....jpeg
      - test.csv
      - train.csv
    - foundation/
      - maybelline.csv
  - model_checkpoints/
    - sam_vit_h_4b8939.pth
    - sam_vit_l_0b3195.pth
  - .gitattributes
  - .gitignore
  - README.md
  - requirements.txt
  - main.ipynb

## Requirements

- Python 3.11.x
- Model Checkpoints from this [Github Repo](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
- Fritzpatrick Dataset from [Kaggle](https://www.kaggle.com/datasets/vinitasilaparasetty/fitzpatrick-classification-by-ethnicity)
