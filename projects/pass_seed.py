import cv2
import numpy as np
 
# [row, col]
NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
 
NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]
 
 
 
def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    return binary_img, points
 
 
 
def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row+offset[0]), rows-1)
                neighbor_col = min(max(0, col+offset[1]), cols-1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    return binary_img
 
# binary_img: bg-0, object-255; int
def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError
 
    binary_img = neighbor_value(binary_img, offsets, False)
    binary_img = neighbor_value(binary_img, offsets, True)
 
    return binary_img
 
 
 
def recursive_seed(binary_img: np.array, seed_row, seed_col, offsets, num, max_num=100):
    rows, cols = binary_img.shape
    binary_img[seed_row][seed_col] = num
    for offset in offsets:
        neighbor_row = min(max(0, seed_row+offset[0]), rows-1)
        neighbor_col = min(max(0, seed_col+offset[1]), cols-1)
        var = binary_img[neighbor_row][neighbor_col]
        if var < max_num:
            continue
        binary_img = recursive_seed(binary_img, neighbor_row, neighbor_col, offsets, num, max_num)
    return binary_img
 
# max_num: max num of NEIGHBOR_HOODS
def Seed_Filling(binary_img, neighbor_hoods, max_num=100):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError
 
    num = 1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var <= max_num:
                continue
            binary_img = recursive_seed(binary_img, row, col, offsets, num, max_num=100)
            num += 1
    return binary_img
 
    
def get_line_mask(binary_img, center, angle):    
    mask=np.zeros(binary_img.shape)
    for x in range(0, binary_img.shape[0]):
        y=np.tan(angle)*(x-center[0])+center[1]
        y=np.round(y)
        
        if y>=binary_img.shape[1]:
            break      
        elif y<0:
            continue
        else:
            mask[int(x),int(y)]=1
        
    return mask    
 
    
 
 
    

capture = cv2.VideoCapture(0)
factor=2

while(True):

    
    # binary_img = np.zeros((4, 7), dtype=np.int16)
    # index = [[0, 2], [0, 5],
    #         [1, 0], [1, 1], [1, 2], [1, 4], [1, 5], [1, 6],
    #         [2, 2], [2, 5],
    #         [3, 1], [3, 2], [3, 4], [3, 6]]
    # for i in index:
    #     binary_img[i[0], i[1]] = np.int16(255)
        
        
        
    # path=r'temp4.jpg'    
    # #frame = cv2.cvtColor(cv2.imread('./assets/my.jpg'), cv2.COLOR_BGR2GRAY)
    
    # frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    
    ret, frame = capture.read()

    width = 200
    top = 100
    bottom = 200
    height = bottom - top


    frame = cv2.resize(frame, (width, width), interpolation=cv2.INTER_CUBIC)

    #frame = frame[top:bottom, 0:width]
    
    frame = frame[top:bottom, width-150:width]
    
    frame=frame[:,::-1]
    
    
    cv2.imshow('frame', cv2.resize(frame, (width*factor, width*factor), interpolation=cv2.INTER_CUBIC))
    
    
    
    
    
    frame_rgb=frame+0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', cv2.resize(gray, (width*factor, width*factor), interpolation=cv2.INTER_CUBIC))

    # canny = cv2.Canny(frame, 100, 300)
    # cv2.imshow('canny', canny)    
    
    
    
    frame=gray
    

    
    
    # plt.imshow(frame)
    # plt.show()
    
    
    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    
    
    
    
    ret, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
    
    mask=frame==0
    frame[mask]=255
    frame[mask==False]=0    
    
    
    binary_img=frame
    
    
    # have fun with traditional image processing algorithms
    kernel = np.ones((5, 5), np.uint8)
    binary_img = cv2.dilate(binary_img, kernel, iterations=3)
    binary_img = cv2.erode(binary_img, kernel, iterations=3)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    binary_img = cv2.copyMakeBorder(binary_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255))
    
    
    binary_img=binary_img[5:-5,5:-5]
        
    
    # cv2.namedWindow('binary_img', cv2.WINDOW_NORMAL)
    # cv2.imshow('binary_img', binary_img)
    # cv2.waitKey(0)
    
    # plt.imshow(binary_img)
    # plt.show()
    
    
     
    print("original binary image")
    print(binary_img)
     
    print("Two_Pass")
    binary_img = Two_Pass(binary_img, NEIGHBOR_HOODS_8)
    binary_img, points = reorganize(binary_img)
    print(binary_img, points)
     
    # print("Seed_Filling")
    # binary_img = Seed_Filling(binary_img, NEIGHBOR_HOODS_8)
    # binary_img, points = reorganize(binary_img)
    # print(binary_img, points)
    
    
    # cv2.namedWindow('binary_img', cv2.WINDOW_NORMAL)
    # cv2.imshow('binary_img', binary_img)
    # cv2.waitKey(0)
    
    cv2.imshow('binary_img', cv2.resize(binary_img, (width*factor, width*factor), interpolation=cv2.INTER_CUBIC)*255/np.max(binary_img))

    
    # plt.imshow(binary_img)
    # plt.show()
    
    
    cluster_num=np.max(np.unique(binary_img))
    
    binary_img_show_center=binary_img+0
    center_flag=cluster_num+1
    
    
    ori_center_list=[]
    area_list=[]
    
    print('')
    for i in range(1,cluster_num+1):
    
        mask=binary_img==i
        area=np.sum(mask)
        points=np.argwhere(binary_img == i)
        center=np.mean(points,0)
        area_list.append(area)
        
        print('cluster %d'%i)
        print('area', area)
        print('center', center)
        print('--------')
        
        center_plot=np.round(center).astype(int)
        ori_center_list.append(center_plot)
        
        binary_img_show_center[center_plot[0]-10:center_plot[0]+10,center_plot[1]-10:center_plot[1]+10]=center_flag
        
        
        #binary_img_show_center[center_plot[0],center_plot[1]]=center_flag
        
    # plt.imshow(binary_img_show_center)
    # plt.show()
    
    ori_center_list=np.array(ori_center_list)
    area_list=np.array(area_list)
    
    rank_idx=np.argsort(area_list)
    rank_idx=rank_idx[::-1]
    #rank_idx=rank_idx[:3]
    
    for rank in rank_idx[3:]:
        rank=rank+1
        binary_img[binary_img==rank]=0
    
    
    
    
    
    # if rank_idx.shape[0]>=3:
    #     cluster_num=3
    # else:
    #     cluster_num=rank_idx.shape[0]

    # rank_idx=rank_idx[:cluster_num]
    # ori_center_list=ori_center_list[rank_idx]
    
    
    
    
    
    
    for i in range(cluster_num):
        cluster_flag=i+1
        center=ori_center_list[i]
    
    
    

        
        
        
        
    mask=get_line_mask(binary_img, np.array([20,50]), np.pi/4)
    
     
    # plt.imshow(mask)
    # plt.show()
    
    
    line_flag=cluster_num+2
    
    
    binary_img_show_line=binary_img+0
    
    
    max_line_mask_list=[]
    max_angle_list=[]
    
    
    for i in range(len(ori_center_list)):
        center=ori_center_list[i]
        cluster_flag=i+1
        
        max_line_len=-1
        max_line_mask=np.zeros((binary_img.shape))
        max_angle=0
        
        rot_num=20
        for j in range(rot_num):
            angle=np.pi/rot_num*j
            line_mask=get_line_mask(binary_img, center, angle)
            
            # plt.imshow(line_mask)
            # plt.show()    
            
            
            #line_len=np.sum(np.logical_and(line_mask, binary_img==cluster_flag))
            #print(line_len)
            
            temp=np.int64(np.logical_and(line_mask, binary_img==cluster_flag))
            # plt.imshow(temp)
            # plt.show()
            
            if np.sum(temp)==0:
                continue
            
            
            points=np.argwhere(temp==1)               
            line_len=np.sum((points.max(0)-points.min(0))**2)            
            
            
            if max_line_len<line_len:
                #print(line_len)
                max_line_len=line_len
                max_line_mask=line_mask
                max_angle=angle
                
        max_line_mask_list.append(max_line_mask)
        max_angle_list.append(angle)
        
    for max_line_mask in max_line_mask_list:   
        binary_img_show_line[max_line_mask==1]=line_flag
        
        
    # plt.imshow(binary_img_show_line)
    # plt.show()    
        
    
    block_size=2
    binary_img_show_line_w_center=binary_img_show_line+0 
    for center in ori_center_list:  
        binary_img_show_line_w_center[center[0]-block_size:center[0]+block_size,center[1]-block_size:center[1]+block_size]=center_flag
            
        
    #cv2.imshow('binary_img_show_line_w_center', binary_img_show_line_w_center*255/np.max(binary_img_show_line_w_center))
        
    # def scalar2rgb_random(scalar):
    #     '''
    #     input
    #     scalar:(n,) numpy
        
    #     output
    #     rgb:(n,3) numpy
    #     '''
        
    #     scalar_set_list=list(set(scalar))
    #     color_map=np.zeros((len(scalar_set_list),3))
    #     for i in range(color_map.shape[0]):
    #         for j in range(color_map.shape[1]):
    #             color_map[i,j]=np.random.randint(0,256)
                
    #     rgb=np.zeros((scalar.shape[0],3))
    #     for s in scalar_set_list:
    #         rgb[scalar==s]=color_map[scalar_set_list.index(s)]
    #     return rgb  
        
    # temp=binary_img_show_line_w_center.reshape((-1,))
    # temp_rgb=scalar2rgb_random(temp)
    
    # temp_rgb=temp_rgb.reshape((*binary_img_show_line_w_center.shape,3))
    
    # cv2.imshow('binary_img_show_line_w_center', temp_rgb/255)
    
    
    for max_line_mask in max_line_mask_list:   
        frame_rgb[max_line_mask==1]=line_flag    
    block_size=2
    binary_img_show_line_w_center=binary_img_show_line+0 
    for center in ori_center_list:  
        frame_rgb[center[0]-block_size:center[0]+block_size,center[1]-block_size:center[1]+block_size]=center_flag
          
        
        
    
    cv2.imshow('frame_rgb', cv2.resize(frame_rgb, (width*factor, width*factor), interpolation=cv2.INTER_CUBIC))
        
    # plt.imshow(binary_img_show_line_w_center)
    # plt.show()    
    cv2.waitKey(1)
    
    
    
    
    
    
    
    

















