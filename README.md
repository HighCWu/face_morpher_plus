# Face Morpher Plus

Mrph human faces plus bsed on facemorpher!\
Scripts will automatically detect frontal faces and skip images if none
is detected.

Built with Python, OpenCV, Numpy, Scipy, Stasm, facemorpher.

Supported on Python 3.6+ and OpenCV \>= 3 (tested with OpenCV 3.4.1)\
Tested on 64bit Linux.

## Requirements

- Install [OpenCV](http://opencv.org): [Mac installation
    steps](https://gist.github.com/alyssaq/f60393545173379e0f3f#file-4-opencv3-with-python3-md)
- Note: OpenCV must be installed either from
    [Homebrew](https://brew.sh) or
    [source](https://github.com/opencv/opencv) as stasm requires the
    library files.
- `pip install -r requirements.txt`

## Use as local command-line utility

    $ git clone https://github.com/alyssaq/face_morpher

## Morphing Faces

Morph from a source to destination image:

    python process.py --src=<src_imgpath> --dest=<dest_imgpath> --plot

All options listed in `morpher.py` (pasted below):

    Morph from source to destination face or
    Morph through all images in a folder

    Usage:
      process.py (--src=<src_path> --dest=<dest_path> | --images=<folder>)
                [--width=<width>] [--height=<height>]
                [--num=<num_frames>] [--fps=<frames_per_second>]
                [--out_frames=<folder>] [--out_video=<filename>]
                [--alpha] [--plot]

    Options:
      -h, --help              Show this screen.
      --src=<src_imgpath>     Filepath to source image (.jpg, .jpeg, .png)
      --dest=<dest_path>      Filepath to destination image (.jpg, .jpeg, .png)
      --images=<folder>       Folderpath to images
      --width=<width>         Custom width of the images/video [default: 500]
      --height=<height>       Custom height of the images/video [default: 600]
      --num=<num_frames>      Number of morph frames [default: 20]
      --fps=<fps>             Number frames per second for the video [default: 10]
      --out_frames=<folder>   Folder path to save all image frames
      --out_video=<filename>  Filename to save a video
      --alpha                 Flag to save transparent background [default: False]
      --plot                  Flag to plot images [default: False]
      --version               Show version.

## Example

    python process.py --src=./example/A.jpg --dest=./example/B.jpg --plot
- In the plot mode, you can move the feature points
  - ![image](example/result/plot_1.png)
- Morph process
  - ![image](example/result/plot_2.png)
  - ![image](example/result.gif)
- From ... to ...
  - ![image](example/result/0.jpg)
  - ![image](example/result/18.jpg)

## License

[MIT](LICENSE)
