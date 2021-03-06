#Network Drawing.py

import numpy as np
from math   import log
from PIL import Image

from pycore.tikzeng import *
from pycore.blocks  import *

px_to_cm = 37.795275591

def f_n(name):
  return '('+name+')'

# Class  : Model_Drawing
# Purpose: High Level Wrapper for tikzeng LaTeX NNetwork Drawing Tool
class Model_Drawing:
  # Member    : __init__
  # Parameters: image - This is a path to an image to write to the file. Image
  #                     size is pulled from here as well for drawing the
  #                     diagrams
  def __init__(self, image, draw = True):
    # Initial layers and setup
    self.arch = [to_head(''),
                 to_cor(),
                 to_begin()]

    # Open and store the size of the image
    im = Image.open(image)
    w, h = im.size


    # Setup of initial values for later
    self.layer_num = 0
    self.cur_layer = None

    # Setup label height and width for display purposes
    self.lab_height= h
    self.lab_width = w

    # Setup figure height and width for drawing purposes
    self.cur_height= h/2
    self.cur_width = w /2
    self.cur_depth = 3

    # Get and add the initial image to the file. NOTE: You cannot add a "from"
    #   field to this.
    if(draw):
      self.add_image(image)
    c,h,w = self.get_params(1)

  # Internal wrapper for readability to increase the layer number
  def increase_layer(self):
    self.layer_num += 1

  # Model    : get_params
  # Arguments: stride - This is the amount of striding done
  #            stride_scale - This is the amount to reduce the striding for
  #                           drawing
  #            unpool - This is if we're deconvoluting, or increasing the dims

  # Internal wrapper to calculate the display and drawing size parameters in
  #   addition to dealing with striding
  def get_params( self, stride = None, stride_scale = .6, unpool = False ):
    self.increase_layer()
    # If there is no stride or stride is 1, don't do striding.
    if stride is not None and stride is not 1:
      # If we stride by different numbers in each dimension, handle it.
      if isinstance(stride,tuple) and len(stride) == 2:
        s_h,s_w = stride
        l_h,l_w = stride
      else:
        s_h,s_w = stride,stride
        l_h,l_w = stride,stride

      # Scale the drawing size to a smaller amount.  This parameter is above.
      s_h     = log(s_h+2)
      s_w     = log(s_w+2)

      # If we're unpooling, invert the striding.
      if unpool:
        l_h = 1 / l_h
        l_w = 1 / l_w
        s_h = 1 / s_h
        s_w = 1 / s_w

      self.cur_height = self.cur_height / s_h
      self.cur_width  = self.cur_width  / s_w
      self.lab_height = self.lab_height / l_h
      self.lab_width  = self.lab_width  / l_w

    # The number of channels is a log scaling since they are powers of 2.
    c = min(log(self.cur_depth),1)

    # The drawing parameters scaled down by a factor of 10.
    h = self.cur_height / 10
    w = self.cur_width  / 10

    return c,h,w

  # Given state size as a 2-tuple, kernel_size as a 3-tuple, and the stride,
  #   add a layer to the network. NOTE: Uses the third value of the kernel_size
  #   in order to store the number of channels in the current number of channels
  #   NOTE: Layer Size is disambiguified. It is (H,W,C) of the LAYER, where the
  #   command is viewing the dimension from the side. This means:
  #                                                               H->H
  #                                                               W->D
  #                                                               D->W
  # By default utilizes log scaling.
  def add_conv( self, kernel, stride = None, offset="(1,0,0)",o_filter="", name = None, to = None, caption = ' ', color = '\\ConvColor' ):
    k_h,k_w,k_c = kernel
    if name is None:
      name = 'conv_%d'%self.layer_num
    c,h,w = self.get_params(stride)

    if to is None :
      to = "%s-east"%self.cur_layer
    if to != '(0,0,0)':
      to = f_n(to)

    # DISAMBIGUATIONS LISTED BELOW.
    self.arch.append( to_Conv( name=name,               # The name of the layer for internal use
                          s_filer=int(self.lab_width),  # Dimension Label
                          # o_filer=int(self.lab_height), # Channel Label
                          n_filer=k_c,                  # Channel Label
                          offset=offset,                # Offset distance from to parameter
                          to=to,                        # The previous state in the network
                          width= c,                     # The depth  of the generated layer
                          height=h,                     # The height of the generated layer
                          depth= w,                     # The width  of the generated layer
                          caption=caption,
                          color= color
                          ) )
    try:
      self.arch.append( to_connection(self.cur_layer, name) )
    except:
      pass
    self.cur_layer = name
    self.cur_depth = k_c

  def add_pool(self, stride = 2, offset="(0,0,0)", name = None, caption = ' ' ):
    if name is None:
      name = 'pool_%d'%self.layer_num
    c,h,w = self.get_params(stride)
    to = "%s-east"%self.cur_layer

    # DISAMBIGUATIONS LISTED BELOW.
    self.arch.append( to_Pool( name=name,              # The name of the layer for internal use
                          # s_filer=self.cur_width,    # Dimension Label
                          # n_filer=self.cur_width,    # Channel Label
                          offset=offset,               # Offset distance from to parameter
                          to=f_n(to),                  # The previous state in the network
                          width=c,                     # The depth  of the generated layer
                          height=h,                    # The height of the generated layer
                          depth=w,                     # The width  of the generated layer
                          caption=caption
                          ) )

    self.cur_layer = name

  def add_unpool_xy( self, kernel, stride = None, offset="(1,0,0)", name = None, to = None, caption = ' ' ):
    if stride is not None and stride is not 1:
      # If we stride by different numbers in each dimension, handle it.
      if isinstance(stride,tuple) is 'tuple' and len(stride) == 2:
        s_h,s_w = stride
        l_h,l_w = stride
      else:
        s_h,s_w = stride,stride
        l_h,l_w = stride,stride

    if name is None:
      name = 'unpool_xy_%d'%self.layer_num
    if to is None:
      to = "%s-east"%self.cur_layer

    stride_x = ( 1,s_w)
    stride_y = (s_h, 1)
    if kernel is None:
      k1 = (3,3,self.cur_depth//2)
      k2 = (3,3,self.cur_depth//4)
    else:
      v1,v2,v3 = kernel
      k1 = (v1,v2,v3)
      k2 = (v1,v2,v3//2)


    cur_size = (self.cur_height,self.cur_width,self.lab_height,self.lab_width)
    offset_z = self.cur_width / 35

    self.add_unpool(k1,stride_x,offset="(3,0, %d)"%offset_z,name=name + '_x' ,to=to) # x
    self.add_unpool(k2,stride_y,offset="(3,0, 0)",name=name + '_xy'      ) # xy

    # Restore saved size.
    self.cur_height,self.cur_width,self.lab_height,self.lab_width = cur_size
    print(self.cur_height,self.cur_width,self.lab_height,self.lab_width)

    self.add_unpool(k1,stride_y,offset="(3,0,-%d)"%offset_z,name=name + '_y' ,to=to) # y
    self.add_unpool(k2,stride_x,offset="(3,0, 0)",name=name + '_yx'      ) # yx

    self.add_conv( kernel, offset="(9,0,0)", name = name + '_comp', to = to ,color='\\FcReluColor')
    self.arch.append( to_connection( name + '_xy',self.cur_layer) )
    self.arch.append( to_connection( name + '_yx',self.cur_layer) )

    self.cur_height,self.cur_width,self.lab_height,self.lab_width = cur_size
    c,h,w = self.get_params(stride, unpool = True )





  def add_unpool( self, kernel, stride = None, offset="(1,0,0)", name = None, to = None, caption = ' ', connection = True ):
    k_h,k_w,self.cur_depth = kernel
    if name is None:
      name = 'unpool_%d'%self.layer_num
    c,h,w = self.get_params( stride, unpool = True )
    if to is None:
      to = "%s-east"%self.cur_layer

    # DISAMBIGUATIONS LISTED BELOW.
    self.arch.append( to_UnPool( name=name,              # The name of the layer for internal use
                            s_filer=int(self.lab_width), # Dimension Label
                            # o_filer=int(self.lab_height),# Height Label
                            n_filer=int(self.cur_depth), # Channel Label
                            offset=offset,               # Offset distance from to parameter
                            to=f_n(to),                  # The previous state in the network
                            width=c,                     # The depth  of the generated layer
                            height=h,                    # The height of the generated layer
                            depth=w,                     # The width  of the generated layer
                            caption=caption
                            ) )
    if connection:
      self.arch.append( to_connection(to, name) )
    self.cur_layer = name

  def add_image(self,image,offset=None,to="(-1,0,0)",opacity=1):
    # The height of images is drawn in cm. 1 unit of TikZ is ~.5 cm, or 5 mm.
    c,h,w = self.get_params()
    if offset is not None:
      to = f_n(to+"-east")
    image = image.split('/')[-1]
    self.arch.append(to_input(image,to=to,width=w*.2,height=h*.2,opacity=opacity))

  def add_skip(self, from_layer, to_layer, pos = 1.25 ):
    print(self.lab_height,self.lab_width)
    self.arch.append( to_skip( of=from_layer, to=to_layer, pos=pos) )

  def add_dense_block(self,kernel,kmap, offset="(1,0,0)", name = None, to = None, caption = ' ' ):
    k_h,k_w,k_c = kernel
    if name is None:
      name = 'dense_%d'%self.layer_num
    c,h,w = self.get_params(1)
    layers = [name+"_%d"%n for n in range(kmap)]
    layers = layers + [name+'_concat']

    # DISAMBIGUATIONS LISTED BELOW.
    for x in range(len(layers)):
      color='\\ConvColor' if x is not len(layers)-1 else '\\SoftmaxColor'
      k_c = self.cur_depth + k_c * kmap if x is len(layers)-1 else k_c
      offset = "(.25,0,0)" if x is not len(layers)-1 else "(.5,0,0)"
      layer = layers[x]
      if to is None :
        to = "%s-east"%self.cur_layer
        # offset = "(1,0,0)"
      if to != '(0,0,0)':
        to = f_n(to)
      self.arch.append( to_Conv( name=layer,             # The name of the layer for internal use
                            s_filer=int(self.lab_width), # Dimension Label
                            n_filer=k_c,                 # Channel Label
                            offset=offset,               # Offset distance from to parameter
                            to=to,                       # The previous state in the network
                            width= c,                    # The depth  of the generated layer
                            height=h,                    # The height of the generated layer
                            depth= w,                    # The width  of the generated layer
                            caption=caption,
                            color=color
                            ) )
      to = layer + '-east'


    self.arch.append(to_dense([self.cur_layer] + layers))

    self.cur_layer = layers[-1]
    self.cur_depth = k_c

  def generate(self,file):
    self.arch.append(to_end())
    to_generate(self.arch, file + '.tex' )
