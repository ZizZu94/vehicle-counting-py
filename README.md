# Vehicle detection py
This project takes a video as input, then detects vehicles frame by frame. After the detection, it is also able to track and count them. For the tracking part, it uses a very simple algorithm; in each frame it calculates the distance between the previous bounding box and the current centroid. In addition, it does also the classification of the vehicles by the area.

## Dependencies

Python

```
$ sudo apt-get install python3 python3-pip
```

Install library [OpenCV](https://pypi.org/project/opencv-python/):

```sh
$ pip install opencv-python
```

## Run

```sh
run main.py
```

## Demo
[Video](https://youtu.be/3RQKkyzUKwQ)
