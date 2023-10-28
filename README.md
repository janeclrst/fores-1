# FoReS (Foundation Recommendation System)

## Folder Structure

Note: Use the following folder structure to run the code.

```
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
```

## Requirements

- Python 3.11.x
- Model Checkpoints from this [Github Repo](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
- Fritzpatrick Dataset from [Kaggle](https://www.kaggle.com/datasets/vinitasilaparasetty/fitzpatrick-classification-by-ethnicity)

## Steps to Run
1. Clone this repo
2. Run `pip install -r requirements.txt`
3. Run the notebook
