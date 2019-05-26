from drawing_tool import Model_Drawing

import os
cwd = os.getcwd()

filename = "Highlevel/DenconvXY"
image    = 'Highlevel/13_0_img.png'
gt       = 'Highlevel/13_0_log.png'

model = Model_Drawing(image,False)
model.get_params(2)
model.add_conv(kernel=(1,1,360),to="(0,0,0)",offset="(0,0,0)")

model.add_unpool_xy(kernel=(3,3,180),stride=2,name="Dest3")

model.generate(filename)
