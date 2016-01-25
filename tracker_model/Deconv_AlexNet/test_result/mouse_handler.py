import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import matplotlib.cm as cm

class MouseHandler(object):
    
    def __init__(self, images, v, vtilde, segment_handler, t, n, N):
        self.images = images
        self.segment_handler = segment_handler
        self.t = t
        self.n = n
        self.N = N
        self.v = v
        self.vtilde = vtilde
        self.segment_list = []
        self.segment_inds = []
        self.match_inds = []
        self.c_seg = -1
        self.c_match = -1        
        self.last_r = -100
        self.last_c = -100
        self.fig1, (self.ax1, self.ax2) = plt.subplots(1, 2, sharey=False)
        self.ax1.set_title('Frame t')
        self.ax1.axis('off')
        self.ax1.axis('off')
        self.ax2.set_title('Frame t+1')        
        self.cid1 = self.fig1.canvas.mpl_connect('button_press_event', self.onclick)
        self.update_images()
        self.fig1.show()
        
#    def update_selected_segment(self):
#        self.c_match = 0
#        self.segment_list, self.segment_inds = self.segment_handler.get_colide_segments(self.t,
#                            self.n, self.last_r, self.last_c)
#        if len(self.segment_list) == 0:
#            self.match_inds = []
#        else:
#            self.match_inds = np.argsort(self.v[self.t+1, self.n, 
#                                        self.segment_inds[self.c_seg], 
#                                        0:int(self.segment_handler.seg_num[self.t+1,self.n])])[::-1]
        
  
    def update_images(self):
        self.im1 = self.ax1.imshow(self.images[self.t * self.N + self.n])
        self.im2 = self.ax2.imshow(self.images[(self.t+1) * self.N + self.n])
        
        cmap = cm.jet
        cmap.set_under('k', alpha=0)
        
        if self.c_seg >= 0 and len(self.segment_list) > 0:
            self.im1 = self.ax1.imshow(self.segment_list[self.c_seg], cmap=cmap,
                         interpolation='none', 
                         clim=[0.9, 1], alpha=.5)
            self.ax1.set_title('Frame t with segment id = %d' % (self.segment_inds[self.c_seg]))
            
            if self.c_match >= 0 and len(self.match_inds) > 0:
                ind = self.match_inds[self.c_match]
                mask = self.segment_handler.get_segment(self.t+1, self.n, ind)
                self.im2 = self.ax2.imshow(mask, cmap=cmap,
                                           interpolation='none',
                                           clim=[0.9, 1], alpha=.5)
                self.ax2.set_title('Frame t+1 with %dth matched (id=%d). Rank = %f' % (self.c_match, ind, self.v[self.t+1, self.n, self.segment_inds[self.c_seg], self.match_inds[self.c_match]]))
        self.fig1.canvas.draw()

    def onclick(self, event):
        #print '\n\nbutton=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
            #event.button, event.x, event.y, event.xdata, event.ydata)
        if event.inaxes is not self.ax1:
            return
        if event.name == 'button_press_event':
            c = int(event.xdata)
            r = int(event.ydata)
            #print 'r=%d c=%d lr=%d lc=%d' % (r, c, self.last_r, self.last_c)
            if event.button == 1:
                #change selected segment
                if r - self.last_r > 0 or c - self.last_c > 0:
                    self.c_seg = 0
                    self.last_r = r
                    self.last_c = c                    
                    self.segment_list, self.segment_inds = self.segment_handler.get_colide_segments(self.t, 
                                            self.n, self.last_r, self.last_c)
                else:
                    self.c_seg = (self.c_seg + 1) % len(self.segment_list)  
                    self.last_r = r
                    self.last_c = c
                self.c_match = 0
                if len(self.segment_list) == 0:
                    self.match_inds = []
                else:
                    self.match_inds = np.argsort(self.v[self.t+1, self.n, 
                                    self.segment_inds[self.c_seg], 
                                    0:int(self.segment_handler.seg_num[self.t+1,self.n])])[::-1]
            elif event.button == 3:
                #show next frame in the ranking
                self.c_match = (self.c_match + 1) % len(self.match_inds)
            self.update_images()
