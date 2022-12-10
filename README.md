# auto-lip-reading
Deep Learning Models for Automated Lip Reading (ALR)

Run `download_grid.py`. It will create a `grid_videos` folder in your home directory and download all the videos to it.

Then run `scripts/extract_mouth_region.py`. It will create a `grid_images` folder in your home directory and populate it with cropped images.

Although `pretrained` weights and config files are present, there is no need to use them.

Model architectures are in `/model_archs`. 
Training scripts are in `/train_scripts`. You can create configs in order to store hyperparameters.