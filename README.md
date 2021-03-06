# ETS-Lane-Detection

This is an object detection algorithm specially design for ETS(Incomplete)
[LaneNet](https://arxiv.org/pdf/1807.01726.pdf)

- [ETS-Lane-Detection](#ets-lane-detection)
  - [Environment](#environment)
  - [Requirements](#requirements)
  - [Test](#test)
  - [Demo](#demo)

## Environment

- OS : Windows
- Python version : 3.7
- Game : Euro Truck Simulator 2

## [Requirements](requirements.txt)

- numpy : 1.16.4
- opencv : 4.1.2
- pillow : 6.1.0
- scipy : 1.3.1
- tensorflow-gpu : 1.13.1
- keras : 2.3.1

## [Test](src/yolo_video.py)

Open the mp4 file in folder named "imgs" by any software.
then run the follow command.

```bash
.../> cd (Path)/ETS-Lane-Detection/src> python yolo_video.py
```
## Demo
<div align="center">
  <img src=demo.gif/>
</div>
