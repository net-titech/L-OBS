from kaffe import Network

# Old version class definition of AlexNet which splits the input into two groups because no enough GPU available
class AlexNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(1000, relu=False, name='fc8')
             .softmax(name='prob'))

# New version class definition of AlexNet which convolves the input normally, without splitting
# class AlexNet(Network):
    # def setup(self):
        # (self.feed('data')
             # .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             # .lrn(2, 2e-05, 0.75, name='norm1')
             # .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             # .conv(5, 5, 256, 1, 1, name='conv2')
             # .lrn(2, 2e-05, 0.75, name='norm2')
             # .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             # .conv(3, 3, 384, 1, 1, name='conv3')
             # .conv(3, 3, 384, 1, 1, name='conv4')
             # .conv(3, 3, 256, 1, 1, name='conv5')
             # .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             # .fc(4096, name='fc6')
             # .fc(4096, name='fc7')
             # .fc(1000, relu=False, name='fc8')
             # .softmax(name='prob'))


# AlexNet Reference
# conv1: 96 11x11-kernels - 3 channels
# conv2: 256 5x5-kernels - 48 channels
# conv3: 384 3x3-kernels - 256 channels
# conv4: 384 3x3-kernels - 192 channels
# conv5: 256 3x3-kernels - 192 channels
# fc6: 4096x9216 matrix
# fc7: 4096x4096 matrix
# fc8: 1000x4096 matrix


# def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, biased=True):
