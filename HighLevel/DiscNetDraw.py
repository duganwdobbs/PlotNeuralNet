from drawing_tool import Model_Drawing

img = 1563

filename = "DiscNetModel"
image    = 'IMG_%d_IMG.png'%img
gt       = 'IMG_%d_ANN.png'%img
sparse   = 'IMG_%d_SPARSE.png'%img

model = Model_Drawing(image)
model.add_conv(kernel=(7,7,16),to="(0,0,0)",offset="(0,0,0)")
model.add_pool(stride=4)
model.add_conv(kernel=(6,6,32))
model.add_pool(stride=4)
model.add_conv(kernel=(5,5,48))
model.add_pool(stride=4)
model.add_conv(kernel=(4,4,64))
model.add_pool(stride=2)
model.add_conv((1,1,1),name="END",caption="Discriminator Logits")
model.add_image(sparse,offset="(0,0,0)",to="END",opacity=.5)


model.generate(filename)
