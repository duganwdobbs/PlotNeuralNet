from drawing_tool import Model_Drawing

import os
cwd = os.getcwd()

filename = "Highlevel/DenseModel"
image    = 'Highlevel/13_0_img.png'
gt       = 'Highlevel/13_0_log.png'

model = Model_Drawing(image)
model.add_conv(kernel=(7,7,12),to="(0,0,0)",offset="(0,0,0)",name="Atrous",color='\\ConvReluColor')
model.add_dense_block(kernel=(3,3,22),kmap=5,name="db1")
model.add_pool(stride=2)
model.add_dense_block(kernel=(3,3,22),kmap=5,name='db2')
model.add_pool(stride=2)
model.add_dense_block(kernel=(3,3,25),kmap=5,name='db3')
model.add_pool(stride=2)

model.add_conv(kernel=(4,4,357),name='Skip4',color='\\FcReluColor')

model.add_unpool(kernel=(3,3,180),stride=2,name="Dest3")
model.add_skip("db3_concat","Dest3",1.75)
model.add_unpool(kernel=(3,3,90),stride=2,name="Dest2")
model.add_skip("db2_concat","Dest2",1.75)
model.add_unpool(kernel=(3,3,45),stride=2,name="Dest1")
model.add_skip("db1_concat","Dest1",1.75)

model.add_conv((1,1,1),name="END2",caption="Prediction")
model.add_image(gt,offset="(0,0,0)",to="END2",opacity=.25)

model.generate(filename)
