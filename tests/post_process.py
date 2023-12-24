import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as pltlines
from pymatreader import read_mat
import cv2

def find_center_orth(directory,row):
    directory = directory_project
    directory_name = row['folder']
    path_snap=directory+directory_name
    path_tile=path_snap+'/Img/TileConfiguration.txt.registered'
    try:
        tileconfig = pd.read_table(path_tile,sep=';',skiprows=4,header=None,converters={2 : ast.literal_eval},skipinitialspace=True)
    except:
        print('error_name')
        path_tile=path_snap+'/Img/TileConfiguration.registered.txt'
        tileconfig = pd.read_table(path_tile,sep=';',skiprows=4,header=None,converters={2 : ast.literal_eval},skipinitialspace=True)
    dirName = path_snap+'/Analysis'
    shape = (3000,4096)
    try:
        os.mkdir(path_snap+'/Analysis')
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    t=time()
    xs =[c[0] for c in tileconfig[2]]
    ys =[c[1] for c in tileconfig[2]]
    dim = (int(np.max(ys)-np.min(ys))+4096,int(np.max(xs)-np.min(xs))+4096)
    ims = []
    for name in tileconfig[0]:
        imname = '/Img/'+name.split('/')[-1]
    #     ims.append(imageio.imread('//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE'+date_plate+plate_str+'/Img/'+name))
        ims.append(imageio.imread(directory+directory_name+imname))
    # contour = scipy.sparse.lil_matrix(dim,dtype=np.uint8)
    # half_circle = scipy.sparse.lil_matrix(dim,dtype=np.uint8)
    contour2 = scipy.sparse.lil_matrix((dim[0]//5,dim[1]//5),dtype=np.uint8)
    half_circle2 = scipy.sparse.lil_matrix((dim[0]//5,dim[1]//5),dtype=np.uint8)
    for index,im in enumerate(ims):
        im_cropped = im
        im_blurred =cv2.blur(im_cropped, (200, 200))
        im_back_rem = (im_cropped+1)/(im_blurred+1)*120
        im_back_rem[im_back_rem>=130]=130
    #     laplacian = cv2.Laplacian((im_cropped<=15).astype(np.uint8),cv2.CV_64F)
    #     points = laplacian>=4
    #     np.save(f'Temp\dilated{tileconfig[0][i]}',dilated)
        boundaries = int(tileconfig[2][index][0]-np.min(xs)),int(tileconfig[2][index][1]-np.min(ys))
    #     contour[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += points
        contour2[boundaries[1]//5:boundaries[1]//5+shape[0]//5,boundaries[0]//5:boundaries[0]//5+shape[1]//5] += cv2.resize((im_blurred<=20).astype(np.uint8),(im.shape[1]//5,im.shape[0]//5))
    #     if index<=80:
    #         half_circle[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += points
    pivot = 3000
    circle = []
    border = []
    for x in range(2000,8000):
        indexes = np.where(contour2[:,x].toarray()>0)[0]
        distances = indexes-pivot
        X = list(indexes)
        Y = list(distances)
        sort_indexes = [x for _, x in sorted(zip(Y, X), key=lambda pair: 1/pair[0])]
        candidates = sort_indexes[0],sort_indexes[-1]
        circle.append((x,min(candidates)))
        border.append((x,max(candidates)))
    array_circ = np.zeros(contour2.shape)
    array_bord = np.zeros(contour2.shape)
    for point in circle:
        array_circ[point[1],point[0]]=1
    for point in border:
        array_bord[point[1],point[0]]=1
    x = np.array(border)[:,1]
    y = np.array(border)[:,0].reshape(-1, 1)

    reg = LinearRegression().fit(y, x)
    orthog = [-1/reg.coef_[0],1]
    orthog = np.array(orthog)/np.linalg.norm(orthog)
    circle_correct=[point for point in circle if point[1]<=3000]
    x = np.array(circle_correct)[:,1]
    y = np.array(circle_correct)[:,0]
    x_m = np.mean(x)
    y_m = np.mean(y)
    method_2 = "leastsq"

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)
    return((xc_2, yc_2),orthog)



class DraggableRectangleCenter:
    lock = None  # only one can be animated at a time

    def __init__(self, rect,drag_orth,dist):
        self.rect = rect
        self.press = None
        self.background = None
        self.drag_orth = drag_orth
        self.circle = None
        self.line = None
        self.dist=dist
    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if (event.inaxes != self.rect.axes
                or DraggableRectangleCenter.lock is not None):
            return
        contains, attrd = self.rect.contains(event)
        if not contains:
            return
        print('event contains', self.rect.xy)
        self.press = self.rect.xy, (event.xdata, event.ydata)
        DraggableRectangleCenter.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        self.rect.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.rect)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if (event.inaxes != self.rect.axes
                or DraggableRectangleCenter.lock is not self):
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.rect)

        # blit just the redrawn area
        canvas.blit(axes.bbox)
        return()

    def on_release(self, event):
        """Clear button press information."""
        if DraggableRectangleCenter.lock is not self:
            return

        self.press = None
        DraggableRectangleCenter.lock = None

        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()
        x,y = self.rect.xy
        if self.circle is not None:
            self.circle.remove()
            self.rect.figure.gca().lines.pop(0)
        x1,y1 = self.drag_orth.rect.xy
        circle = plt.Circle((x,y),1000,alpha = 0.3,color= 'red')
        direction = np.array((x1-x,y1-y))
        direction = direction/np.linalg.norm(direction)
        pos_line = np.array((x,y))+self.dist*direction
        orth_direct = np.array([direction[1],-direction[0]])
        extension = 1000
        deb_line = pos_line + extension*orth_direct
        end_line = pos_line - extension*orth_direct
        line = pltlines.Line2D((deb_line[0],end_line[0]),(deb_line[1],end_line[1]),color='red')
        self.line = line
        self.circle = circle
        ax = self.rect.figure.gca()
        ax.add_patch(circle)
        ax.add_line(line)
    def disconnect(self):
        """Disconnect all callbacks."""
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

class DraggableRectangleOrth:
    lock = None  # only one can be animated at a time

    def __init__(self, rect):
        self.rect = rect
        self.press = None
        self.background = None
        self.line = None
    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if (event.inaxes != self.rect.axes
                or DraggableRectangleOrth.lock is not None):
            return
        contains, attrd = self.rect.contains(event)
        if not contains:
            return
        print('event contains', self.rect.xy)
        self.press = self.rect.xy, (event.xdata, event.ydata)
        DraggableRectangleOrth.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        self.rect.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.rect)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if (event.inaxes != self.rect.axes
                or DraggableRectangleOrth.lock is not self):
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.rect)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        """Clear button press information."""
        if DraggableRectangleOrth.lock is not self:
            return

        self.press = None
        DraggableRectangleOrth.lock = None

        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()
    def disconnect(self):
        """Disconnect all callbacks."""
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

