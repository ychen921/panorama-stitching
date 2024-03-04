# panorama-stitching

## Data

## Usage
To stitch images, you can run the following command.
```
python3 ./Wrapper.py --BasePath {Path to your dataset}
```

You can use `--Dataset` to set the training set or testing set, and also `--Set` to select which subset of images will be stitched.
```
python3 ./Wrapper.py --BasePath {Path to your dataset} --Dataset Train --Set Set1
```

## Visualization