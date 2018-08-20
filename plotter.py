"""
Plot and save images
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
import numpy as np
import cv2

def bgr2rgb(img):
  # OpenCV's BGR to RGB
  rgb = np.copy(img)
  rgb[..., 0], rgb[..., 2] = img[..., 2], img[..., 0]
  return rgb

def check_do_plot(func):
  def inner(self, *args, **kwargs):
    if self.do_plot:
      func(self, *args, **kwargs)

  return inner

def check_do_save(func):
  def inner(self, *args, **kwargs):
    if self.do_save:
      func(self, *args, **kwargs)

  return inner

class Plotter(object):
  def __init__(self, plot=True, rows=0, cols=0, num_images=0, out_folder=None, out_filename=None, label=False):
    self.save_counter = 1
    self.plot_counter = 1
    self.do_plot = plot
    self.do_save = out_filename is not None
    self.out_filename = out_filename
    self.set_filepath(out_folder)

    if (rows + cols) == 0 and num_images > 0:
      # Auto-calculate the number of rows and cols for the figure
      self.rows = np.ceil(np.sqrt(num_images / 2.0))
      self.cols = np.ceil(num_images / self.rows)
    else:
      self.rows = rows
      self.cols = cols
    
    if label:
      self.label = True
      self.fig = []
      self.on_press = []
      self.on_release = []
      self.on_motion = []
    else:
      self.label = False

  def set_filepath(self, folder):
    if folder is None:
      self.filepath = None
      return

    if not os.path.exists(folder):
      os.makedirs(folder)
    self.filepath = os.path.join(folder, 'frame{0:03d}.png')
    self.do_save = True

  @check_do_save
  def save(self, img, filename=None):
    if self.filepath:
      filename = self.filepath.format(self.save_counter)
      self.save_counter += 1
    elif filename is None:
      filename = self.out_filename

    mpimg.imsave(filename, bgr2rgb(img))
    print(filename + ' saved')

  @check_do_plot
  def plot_one(self, img, pts=None):
    p = plt.subplot(self.rows, self.cols, self.plot_counter)
    p.axes.get_xaxis().set_visible(False)
    p.axes.get_yaxis().set_visible(False)
    plt.imshow(bgr2rgb(img))
    self.plot_counter += 1
    if self.label:
      if not type(pts) == type(np.array([0])):
        raise Exception('Need Points')
      n=np.arange(pts.shape[0])
      for i,txt in enumerate(n):
        cicle = plt.Circle((pts[i][0],pts[i][1]), 7, color = 'r', alpha = 0.5, label = txt)
        p.axes.add_patch(cicle)
        dc = DraggableCircle(cicle, pts, i)
        self.on_press.append(dc.on_press)
        self.on_release.append(dc.on_release)
        self.on_motion.append(dc.on_motion)
      fig = p.figure
      fig.canvas.mpl_connect('button_press_event', self.on_mousedown)
      fig.canvas.mpl_connect('motion_notify_event', self.on_mousemove)
      fig.canvas.mpl_connect('button_release_event', self.on_mouseup)

  @check_do_plot
  def show(self):
    plt.gcf().subplots_adjust(hspace=0.05, wspace=0,
                              left=0, bottom=0, right=1, top=0.98)
    plt.axis('off')
    plt.show()

  @check_do_plot
  def plot_mesh(self, points, tri, color='k'):
    """ plot triangles """
    for tri_indices in tri.simplices:
      t_ext = [tri_indices[0], tri_indices[1], tri_indices[2], tri_indices[0]]
      plt.plot(points[t_ext, 0], points[t_ext, 1], color)
  
  def on_mousedown(self, event):
    for i in self.on_press:
      i(event)
  def on_mouseup(self, event):
    for i in self.on_release:
      i(event)
  def on_mousemove(self, event):
    for i in self.on_motion:
      i(event)

class DraggableCircle:
  lock = None  # only one can be animated at a time
  def __init__(self, cicle, pts, index):
    self.cicle = cicle
    self.pts = pts
    self.index = index
    self.press = None
    self.background = None

  def connect(self):
    'connect to all the events we need'
    self.cidpress = self.cicle.figure.canvas.mpl_connect('button_press_event', self.on_press)
    self.cidrelease = self.cicle.figure.canvas.mpl_connect('button_release_event', self.on_release)
    self.cidmotion = self.cicle.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

  def on_press(self, event):
    'on button press we will see if the mouse is over us and store some data'
    if event.inaxes != self.cicle.axes: return
    if DraggableCircle.lock is not None: return
    contains, attrd = self.cicle.contains(event)
    if not contains: return
    # print('event contains', self.cicle.center)
    x0, y0 = self.cicle.center
    self.press = x0, y0, event.xdata, event.ydata
    DraggableCircle.lock = self

    # draw everything but the selected Circle and store the pixel buffer
    canvas = self.cicle.figure.canvas
    axes = self.cicle.axes
    self.cicle.set_animated(True)
    canvas.draw()
    self.background = canvas.copy_from_bbox(self.cicle.axes.bbox)

    # now redraw just the Circle
    axes.draw_artist(self.cicle)

    # and blit just the redrawn area
    canvas.blit(axes.bbox)

  def on_motion(self, event):
    'on motion we will move the cicle if the mouse is over us'
    if DraggableCircle.lock is not self:
      return
    if event.inaxes != self.cicle.axes: return
    x0, y0, xpress, ypress = self.press
    dx = event.xdata - xpress
    dy = event.ydata - ypress
    self.cicle.center = (x0+dx, y0+dy)
    self.pts[self.index][0] = round(x0+dx)
    self.pts[self.index][1] = round(y0+dy)
    canvas = self.cicle.figure.canvas
    axes = self.cicle.axes
    # restore the background region
    canvas.restore_region(self.background)

    # redraw just the current Circle
    axes.draw_artist(self.cicle)

    # blit just the redrawn area
    canvas.blit(axes.bbox)

  def on_release(self, event):
    'on release we reset the press data'
    if DraggableCircle.lock is not self:
      return

    self.press = None
    DraggableCircle.lock = None

    # turn off the cicle animation property and reset the background
    self.cicle.set_animated(False)
    self.background = None

    # redraw the full figure
    self.cicle.figure.canvas.draw()

  def disconnect(self):
    'disconnect all the stored connection ids'
    self.cicle.figure.canvas.mpl_disconnect(self.cidpress)
    self.cicle.figure.canvas.mpl_disconnect(self.cidrelease)
    self.cicle.figure.canvas.mpl_disconnect(self.cidmotion)