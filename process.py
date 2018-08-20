"""
::

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
"""
from docopt import docopt
import os
import numpy as np
import cv2

from facemorpher import locator
from facemorpher import aligner
from facemorpher import warper
from facemorpher import blender
import plotter
from facemorpher import videoer

def verify_args(args):
  if args['--images'] is None:
    valid = os.path.isfile(args['--src']) & os.path.isfile(args['--dest'])
    if not valid:
      print('--src=%s or --dest=%s file does not exist. Double check the supplied paths' % (
        args['--src'], args['--dest']))
      exit(1)
  else:
    valid = os.path.isdir(args['--images'])
    if not valid:
      print('--images=%s is not a valid directory' % args['--images'])
      exit(1)

def calc_angle(point_s,point_e):
  import math
  angle=0
  x_point_s = point_s[0]
  y_point_s = point_s[1]
  x_point_e = point_e[0]
  y_point_e = point_e[1]
  y_se= y_point_e-y_point_s
  x_se= x_point_e-x_point_s
  if x_se==0 and y_se>0:
      angle = 360
  if x_se==0 and y_se<0:
      angle = 180
  if y_se==0 and x_se>0:
      angle = 90
  if y_se==0 and x_se<0:
      angle = 270
  if x_se>0 and y_se>0:
      angle = math.atan(x_se/y_se)*180/math.pi
  elif x_se<0 and y_se>0:
      angle = 360 + math.atan(x_se/y_se)*180/math.pi
  elif x_se<0 and y_se<0:
      angle = 180 + math.atan(x_se/y_se)*180/math.pi
  elif x_se>0 and y_se<0:
      angle = 180 + math.atan(x_se/y_se)*180/math.pi
  return angle

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def load_image_points(path, size):
  img = cv2.imread(path)
  points = locator.face_points(img)
  if len(points) == 0:
    print('No face in %s' % path)
    return None, None
  else:
    center_fore_head = points[14]
    center_tip_of_chin = points[6]
    angle = calc_angle(center_fore_head, center_tip_of_chin)
    img = rotate(img, -angle, ((center_fore_head[0]+center_tip_of_chin[0])//2, (center_fore_head[1]+center_tip_of_chin[1])//2))
    points = locator.face_points(img)
    return aligner.resize_align(img, points, size)

def load_valid_image_points(imgpaths, size):
  for path in imgpaths:
    img, points = load_image_points(path, size)
    if img is not None:
      print(path)
      yield (img, points)

def list_imgpaths(images_folder=None, src_image=None, dest_image=None):
  if images_folder is None:
    yield src_image
    yield dest_image
  else:
    for fname in os.listdir(images_folder):
      if (fname.lower().endswith('.jpg') or
         fname.lower().endswith('.png') or
         fname.lower().endswith('.jpeg')):
        yield os.path.join(images_folder, fname)

def alpha_image(img, points):
  mask = blender.mask_from_points(img.shape[:2], points)
  return np.dstack((img, mask))

def correct_colours(src_img_with_alpha, dest_img_with_alpha, points_matrix):
  src_img = src_img_with_alpha[:,:,:3]*0.95
  dest_img = dest_img_with_alpha[:,:,:3]*0.95
  COLOUR_CORRECT_BLUR_FRAC = 0.6
  LEFT_EYE_POINTS =  list(range(29, 39))
  RIGHT_EYE_POINTS = list(range(28, 29)) + list(range(39, 48))
  blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                                          np.mean(points_matrix[LEFT_EYE_POINTS], axis=0) -
                                          np.mean(points_matrix[RIGHT_EYE_POINTS], axis=0))
  blur_amount = int(blur_amount)
  if blur_amount % 2 == 0:
    blur_amount += 1
  src_img_blur = cv2.GaussianBlur(src_img, (blur_amount, blur_amount), 0).astype(np.float64)
  dest_img_blur = cv2.GaussianBlur(dest_img, (blur_amount, blur_amount), 0).astype(np.float64)

  # Avoid divide-by-zero errors.
  dest_img_blur += 1 * (dest_img_blur <= 1.0)

  result_img = dest_img * src_img_blur / dest_img_blur
  overflow_point = result_img / 0.95 > 255
  result_img[overflow_point] = 255
  result_img[~overflow_point] /= 0.95

  return result_img.astype(np.uint8)


def process_edge(src_img, dest_img, width=500, height=600):
  BLUR_RADIUS = [int((width - width/1.05)/2),int((height - height/1.05)/2)]
  for index in range(0, 2):
    if BLUR_RADIUS[index] % 2 == 0:
      BLUR_RADIUS[index] += 1
  BLUR_RADIUS = tuple(BLUR_RADIUS)

  from PIL import Image
  img_canvas = Image.new('L', (width,height))
  img = Image.fromarray(np.uint8(dest_img[:,:,-1])).resize((int(width/1.1), int(height/1.1)))
  img_canvas.paste(img, BLUR_RADIUS)
  img = np.asarray(img_canvas)
  img.flags.writeable = True
  img = cv2.GaussianBlur(img, BLUR_RADIUS, 0)
  img = np.array([img, img, img]).transpose((1, 2, 0)).astype(np.float64)
  result_img = src_img[:,:,:3].copy().astype(np.float64)
  result_img[img > 0] = src_img[:,:,:3][img > 0]*(1 - img[img > 0]/255) + dest_img[:,:,:3][img > 0]*(img[img > 0]/255)
  result_img = result_img.astype(np.uint8)
  return result_img

def morph(src_img, src_points, dest_img, dest_points,
          video, width=500, height=600, num_frames=20, fps=10,
          out_frames=None, out_video=None, alpha=False, plot=False):
  """
  Create a morph sequence from source to destination image
  :param src_img: ndarray source image
  :param src_img: source image array of x,y face points
  :param dest_img: ndarray destination image
  :param dest_img: destination image array of x,y face points
  :param video: facemorpher.videoer.Video object
  """
  size = (height, width)
  stall_frames = np.clip(int(fps*0.15), 1, fps)  # Show first & last longer
  plt = plotter.Plotter(plot, num_images=num_frames, out_folder=out_frames)
  num_frames -= (stall_frames * 2)  # No need to process src and dest image
  label = plotter.Plotter(plot, num_images=2, out_folder=out_frames, label=True)
  label.plot_one(src_img, src_points)
  label.plot_one(dest_img, dest_points)
  label.show()
  plt.plot_one(src_img)
  video.write(src_img, 1)
  try:
    os.mkdir(os.path.join(os.getcwd(),'result'))
    os.mkdir(os.path.join(os.getcwd(),'result','src'))
    os.mkdir(os.path.join(os.getcwd(),'result','src_corners'))
    os.mkdir(os.path.join(os.getcwd(),'result','end'))
    os.mkdir(os.path.join(os.getcwd(),'result','average'))
  except Exception as e:
    print(e)

  # Produce morph frames!
  for percent in np.linspace(1, 0, num=num_frames):
    points = locator.weighted_average_points(src_points, dest_points, percent)
    src_face = warper.warp_image(src_img, src_points, points, size)
    end_face = warper.warp_image(dest_img, dest_points, points, size)
    average_face = blender.weighted_average(src_face, end_face, percent)
    average_face = alpha_image(average_face, points) if alpha else average_face
    average_face[:,:,:3] = correct_colours(src_face, average_face, np.matrix(points))
    corners = np.array([np.array([0,0]),np.array([0,height-2]),np.array([width-2,0]),np.array([width-2,height-2])])
    src_points_with_corners = np.concatenate((src_points, corners))
    points_with_corners = np.concatenate((points, corners))
    src_face_corners = warper.warp_image(src_img, src_points_with_corners, points_with_corners, size)
    average_face = process_edge(src_face_corners, average_face, width, height)
    plt.plot_one(average_face)
    filename = '%d.jpg' % int((1-percent)*num_frames)
    cv2.imwrite(os.path.join(os.getcwd(),'result','src',filename), src_face)
    cv2.imwrite(os.path.join(os.getcwd(),'result','src_corners',filename), src_face_corners)
    cv2.imwrite(os.path.join(os.getcwd(),'result','end',filename), end_face)
    cv2.imwrite(os.path.join(os.getcwd(),'result','average',filename), average_face)
    plt.save(average_face)
    video.write(average_face)

  plt.plot_one(dest_img)
  video.write(dest_img, stall_frames)
  plt.show()

def morpher(imgpaths, width=500, height=600, num_frames=20, fps=10,
            out_frames=None, out_video=None, alpha=False, plot=False):
  """
  Create a morph sequence from multiple images in imgpaths
  :param imgpaths: array or generator of image paths
  """
  video = videoer.Video(out_video, fps, width, height)
  images_points_gen = load_valid_image_points(imgpaths, (height, width))
  src_img, src_points = next(images_points_gen)
  for dest_img, dest_points in images_points_gen:
    morph(src_img, src_points, dest_img, dest_points, video,
          width, height, num_frames, fps, out_frames, out_video, alpha, plot)
    src_img, src_points = dest_img, dest_points
  video.end()

def main():
  args = docopt(__doc__, version='Face Morpher Plus 1.0')
  verify_args(args)
  args.alpha = True

  morpher(list_imgpaths(args['--images'], args['--src'], args['--dest']),
          int(args['--width']), int(args['--height']),
          int(args['--num']), int(args['--fps']),
          args['--out_frames'], args['--out_video'],
          args['--alpha'], args['--plot'])


if __name__ == "__main__":
  main()