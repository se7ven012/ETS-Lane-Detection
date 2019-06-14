# ETS-Lane-Detection

This is a lane detection algorithm specially design for ETS

- [ETS-Lane-Detection](#ets-lane-detection)
  - [Environment](#environment)
  - [Requirements](#requirements)
    - [For Windows](#for-windows)
    - [For Linux](#for-linux)
  - [Test](#test)

## Environment

- OS : Windows / Linux(incomplete)
- Python version : 3.x
- Game : Euro Truck Simulator 2
- opencv : 4.0.x

## [Requirements](requirements.txt)

### For Windows

- numpy
- cv2
- PIL
- win32gui
- pywin32

### For Linux

use two methods to get the interest image.
the first one, use the xlib package and need install these following packages.

- python-xlib
- ewmh

the other one, use the third-party package called 'maim' and it [Installation](https://github.com/naelstrof/maim).

- maim

## [Test](src/test/test.py)

Open the mp4 file in folder named "imgs" by any software.
then run the follow command.

```bash
.../> cd (Path)/ETS-Lane-Detection/src/test
.../ETS-Lane-Detection/src/test> python3 test.py
```
