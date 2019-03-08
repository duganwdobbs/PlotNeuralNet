from drawing_tool import Model_Drawing

img = 1563

filename = "DisCountNetModel"
image    = 'IMG_%d_IMG.png'%img
gt       = 'IMG_%d_ANN.png'%img
sparse   = 'IMG_%d_SPARSE.png'%img

model = Model_Drawing(image)
model.add_conv((7,7,16),to="(0,0,0)",offset="(0,0,0)",name="Skip1")
model.add_pool(4)
model.add_conv((6,6,32))
model.add_pool(4)
model.add_conv((5,5,48))
model.add_pool(4)
model.add_conv((4,4,64))
model.add_pool(2)
model.add_conv((1,1,1),name='END')
model.add_image(sparse,offset="(0,0,0)",to="END",opacity=.5)

# Resize the network.
model.get_params(4,unpool=True)
model.get_params(4,unpool=True)
model.get_params(4,unpool=True)
model.get_params(2,unpool=True)

model.add_conv(kernel=(1,1,3),name="Sparse",offset="(3,0,0)",caption="Sparse Image")
model.add_image(sparse,offset="(0,0,0)",to="Sparse",opacity=.5)


model.add_conv(kernel=(3,3,16),offset="(2,0,0)",name="Skip1")
model.add_pool(stride=2)
model.add_conv(kernel=(5,5,32),name='Skip2')
model.add_pool(stride=2)
model.add_conv(kernel=(5,5,48),name='Skip3')
model.add_pool(stride=2)
model.add_conv(kernel=(4,4,64),name='Skip4')
model.add_pool(stride=2)
model.add_conv(kernel=(1,1,80))
model.add_unpool(kernel=(3,3,64),stride=2,name="Dest4")
model.add_skip("Skip4","Dest4")
model.add_unpool(kernel=(3,3,32),stride=2,name="Dest3")
model.add_skip("Skip3","Dest3")
model.add_unpool(kernel=(3,3,16),stride=2,name="Dest2")
model.add_skip("Skip2","Dest2")
model.add_unpool(kernel=(3,3,8),stride=2,name="Dest1")
model.add_skip("Skip1","Dest1")

model.add_conv((1,1,1),name="END2",caption="Probability Heat Map")
model.add_image(gt,offset="(0,0,0)",to="END2",opacity=.5)

model.generate(filename)
