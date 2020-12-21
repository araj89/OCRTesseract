import numpy as np
import pandas as pd
import cv2
import imutils
from imutils import perspective
import math
import pytesseract
from locale import atoi
from tqdm import tqdm, trange
import os
import timeit
import datetime

debug_mode = False
img_path = "../imgs/1-4500/1221-1240/005.jpg"




digit_config = r'--oem 3 --psm 7 outputbase digits'
text_config = r'--oem 3 --psm 7'
date_config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/'
#date_config = r'--oem 3 --psm 7 outputbase digits'

THRESOCR_DEL = 30
OCR_IMG_TYPE = False

# don't edit this global vars
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
CheckType = 0
TableType = -1

def get_index_x(img,curx, min_dist = 12):
    sy = []
    ey = []

    bfirst = False
    ci = 0
    proc_black = False
    for i in range(img.shape[0]):
        v = img[i, curx]
        if v > 128:
            bfirst = True
            ci = 0
        if not bfirst:
            continue

        if v == 0 and proc_black == False:
            proc_black = True
            sy.append(i)
        if v > 128 and proc_black == True and i - sy[-1] > min_dist:
            proc_black = False
            ey.append(i-1)
            ci += 1

    # last exception
    if len(sy) > len(ey):
        ey.append(img.shape[0]-1)
    return sy, ey

def get_index_y(img, cury, min_dist = 15):
    sx = []
    ex = []

    bfirst = False
    proc_black = False
    ci = 0
    for i in range(img.shape[1]):
        v = img[cury, i]
        if v > 128:
            bfirst = True
            ci = 0
        if not bfirst:
            continue

        if v == 0 and proc_black == False:
            proc_black = True
            sx.append(i + 2)
        if v > 128 and proc_black == True and i - sx[-1] > min_dist:
            proc_black = False
            ex.append(i - 2)
            ci += 1

    if len(sx) > len(ex):
        ex.append(img.shape[1] - 1)
    if len(ex) > len(sx):
        ex = ex[:-1]
    return sx, ex

def bbox(img2):
    h, w = img2.shape
    sum = img2.sum()
    if sum == 0:
        return 0, h-1, 0, w-1
    rows = np.any(img2, axis=1)
    cols = np.any(img2, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax+1, cmin, cmax+1

def bbox_noise_remove(img, area_thres = 8):
    h, w = img.shape
    if h < 2 or w < 2:
        return img
    # hroizontal line remove
    kernel = np.ones((1, h + 2), np.uint8)
    limg = cv2.erode(img, kernel, None, iterations=1)
    limg = cv2.dilate(limg, kernel, None, iterations=1)

    if h > 6:
        limg[3:h-3] = 0

    res = cv2.subtract(img, limg)

    if False:
        cv2.imshow('img', img)
        cv2.imshow('limg', limg)
        cv2.imshow('removed', res)
        cv2.waitKey(0)

    cnts = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ncnts = []
    for cnt in cnts:
        # remove cnts that has small area
        if cv2.contourArea(cnt) < area_thres:
            cv2.drawContours(res, [cnt], -1, 0, -1)
            continue
        # remove cnts that has small height
        box = cv2.boundingRect(cnt)
        if box[3] < 3:
            cv2.drawContours(res, [cnt], -1, 0, -1)
            continue

        ncnts.append((cnt, box[0], box[0] + box[2]))

    ncnts = sorted(ncnts, key = (lambda x: x[1]), reverse=False)
    if len(ncnts) > 2:
        d1 = ncnts[1][1] - ncnts[0][2]
        d2 = ncnts[-1][1] - ncnts[-2][2]
        if  d1 > h:
            cv2.drawContours(res, [ncnts[0][0]], -1, 0, -1)
        if  d2 > h:
            cv2.drawContours(res, [ncnts[-1][0]], -1, 0, -1)

    if False:
        cv2.imshow('1', img)
        cv2.imshow('2', res)
        cv2.waitKey(0)

    return res

def get_tbl_type(tbl_pts, line_delta = 15):
    if len(tbl_pts) < 13:
        return -1

    pt0 = np.array([(tbl_pts[0][0] + tbl_pts[0][2]), (tbl_pts[0][1] + tbl_pts[0][3])]) / 2
    pt1 = np.array([(tbl_pts[1][0] + tbl_pts[1][2]), (tbl_pts[1][1] + tbl_pts[1][3])]) / 2
    dist_c = math.sqrt((pt0[0] - pt1[0]) * (pt0[0] - pt1[0]) + (pt0[1] - pt1[1]) * (pt0[1] - pt1[1]))
    cnt = 2
    while True:
        pt0 = pt1
        pt1 =  np.array([(tbl_pts[cnt][0] + tbl_pts[cnt][2]), (tbl_pts[cnt][1] + tbl_pts[cnt][3])]) / 2
        dist = math.sqrt((pt0[0] - pt1[0]) * (pt0[0] - pt1[0]) + (pt0[1] - pt1[1]) * (pt0[1] - pt1[1]))
        if math.fabs(dist - dist_c) > line_delta:
            break
        if cnt == len(tbl_pts) - 1:
            cnt = cnt + 1
            break
        cnt = cnt + 1
    cnt = cnt - 1

    w = math.sqrt((tbl_pts[0][0] - tbl_pts[0][2]) * (tbl_pts[0][0] - tbl_pts[0][2]) + (tbl_pts[0][1] - tbl_pts[0][3]) * (tbl_pts[0][1] - tbl_pts[0][3]))
    w_thres = w / 3
    if tbl_pts[cnt][1] - tbl_pts[0][1] > w_thres or tbl_pts[cnt][3] - tbl_pts[0][3] > w_thres:
        return -1
    return cnt

def get_tbl_info(img, t, ncut, base_threshold = 15):
    thresholds = [base_threshold, 2 * base_threshold, 3 * base_threshold]
    tbl_types = []

    img_smooth = cv2.GaussianBlur(img, (3, 3), 0)

    end_pts_s = []
    for t2 in thresholds:
        _, thres_img = cv2.threshold(img_smooth, t + t2, 255, cv2.THRESH_BINARY_INV)
        thres_img[:, 0:ncut] = 0
        thres_img[:, -ncut:] = 0
        thres_img[0:ncut, :] = 0
        thres_img[-ncut:, :] = 0

        h, w = thres_img.shape
        kernel = np.ones((1, 17), np.uint8)
        eximg = cv2.erode(thres_img, kernel, None, iterations=1)
        eximg = cv2.dilate(eximg, kernel, None, iterations=1)

        cnts = cv2.findContours(eximg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        end_pts = []
        for cnt in cnts:
            sx, ex, sy, ey = 9999, 0, 0, 0
            for pt in cnt:
                pt = pt[0]
                if pt[0] < sx:
                    sx = pt[0]
                    sy = pt[1]
                if pt[0] > ex:
                    ex = pt[0]
                    ey = pt[1]
            if (ex - sx) < w *4/7:
                continue
            end_pts.append([sx,sy, ex, ey])

        end_pts = sorted(end_pts, key = (lambda pts: (pts[1] + pts[3])), reverse=False)

        if debug_mode:
            cnt = 0
            for end_pt in end_pts:
                cv2.circle(eximg, (end_pt[0], end_pt[1]), 20, 255, 2)
                cv2.circle(eximg, (end_pt[2], end_pt[3]), 20, 255, 2)
                cnt += 1
            cv2.imshow('ttt', eximg)
            cv2.imshow('thres_tbl', thres_img)
            cv2.waitKey(0)

        nlines = get_tbl_type(end_pts)
        tbl_types.append(nlines)
        end_pts_s.append(end_pts)

    global TableType
    mmax = -2
    id = 0
    cc = 0
    for each in tbl_types:
        if mmax < each:
            mmax = each
            id = cc
        cc = cc + 1
    end_pts = end_pts_s[id]
    TableType = mmax

    if TableType != 12 and TableType != 14 and TableType != 15:
        print ('new table type')
        TableType = -1
    if TableType == -1:
        return [], [], [], []

    return np.array([end_pts[0][0], end_pts[0][1]]), np.array([end_pts[0][2], end_pts[0][3]]), np.array([end_pts[TableType][2], end_pts[TableType][3]]), np.array([
        end_pts[TableType][0], end_pts[TableType][1]])

def refine_row(img_vh, h1, h2, bd, dbd,sp, ep, cols):
    last_margine_thres = 150
    h, w = img_vh.shape
    flgs = [False, False, False, False, False]

    for i in range(len(sp) - 1):
        c = ep[i]
        f = False
        for j in range(len(cols)):
            col = cols[j]
            if math.fabs(c - col) < dbd:
                if flgs[j] == True:
                    f = False
                else:
                    f = True
                flgs[j] = True
                break

        if not f and not ((w - ep[i]) < last_margine_thres and i >= len(ep) - 1) :
            img_vh[h1:h2, ep[i]:sp[i+1]+1] = 0

    for i in range(len(cols)):
        if flgs[i]:
            continue
        img_vh[h1:h2, cols[i]-bd:cols[i]+bd+1] = 255
    return img_vh

def refine_vh12(img_vh, cols):
    bd = 3
    dbd = 40
    delta = 2
    delta2 = 2

    # paint white value in border
    img_vh[:, 0:6] = 255
    img_vh[0:6, :] = 255

    cols_sp, cols_ep = get_index_x(img_vh, int((cols[1] + cols[2]) / 2))
    if len(cols_sp) < 12 or len(cols_ep) < 12:
        return []

    h01, h02 = cols_sp[0], cols_ep[0]
    t = int((h01 + h02) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    t = 0
    if len(sp) > 1 and sp[-1] > img_vh.shape[1] - 200:
        t = sp[-1]
    if t > 0:
        img_vh = img_vh[:, :t-3]

    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2
    h21, h22 = cols_sp[2] + delta, cols_ep[2] + delta2
    h61, h62, h71, h72 = cols_sp[4] + delta, cols_ep[4] + delta2, cols_sp[5] + delta, cols_ep[5] + delta2
    h91, h92 = cols_sp[11] + delta, cols_ep[11] + delta2

    t = int((h11 + h12) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist = 0)
    img_vh = refine_row(img_vh, h11, h12, bd, dbd, sp, ep, [cols[0]])

    t = int((h21 + h22) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist = 0)
    img_vh = refine_row(img_vh, h21, h22, bd, dbd, sp, ep, [cols[0]])

    t = int((h61 + h62) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist = 0)
    img_vh = refine_row(img_vh, h61, h62, bd, dbd, sp, ep, [cols[1], cols[2], cols[3], cols[4]])

    t = int((h71 + h72) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist = 0)
    img_vh = refine_row(img_vh, h71, h72, bd, dbd, sp, ep, [cols[1], cols[2], cols[3], cols[4]])

    t = int((h91 + h92) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist = 0)
    img_vh = refine_row(img_vh, h91, h92, bd, dbd, sp, ep, [cols[1], cols[2], cols[3], cols[4]])

    return img_vh
def get_ncell_12(img_table_org):
    cols = np.array([0.2814, 0.3872, 0.5311, 0.8188, 0.8654])

    h, w = img_table_org.shape
    cols = cols * w
    cols = cols.astype(np.int)

    dst_img_table = cv2.GaussianBlur(img_table_org, (5, 5), 0)
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    img_smooth = cv2.filter2D(dst_img_table, -1, kernel_sharpening)

    t, _ = cv2.threshold(img_table_org, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, img_cell = cv2.threshold(img_table_org, t + 30, 255, cv2.THRESH_BINARY_INV)
    t = max(30, t - THRESOCR_DEL)
    _, img_table = cv2.threshold(img_smooth, t, 255, cv2.THRESH_BINARY_INV)

    delta = 1
    delta2 = 0
    kernel_len_v = img_cell.shape[0] // 10
    kernel_len_h = img_cell.shape[1] // 10
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_v))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_h, 1))

    v_lines = cv2.erode(img_cell, ver_kernel, iterations=1)
    v_lines = cv2.dilate(v_lines, ver_kernel, iterations=1)
    h_lines = cv2.erode(img_cell, hor_kernel, iterations=1)
    h_lines = cv2.dilate(h_lines, hor_kernel, iterations=1)
    img_vh = cv2.addWeighted(v_lines, 1.0, h_lines, 1.0, 0.0)

    if debug_mode:
        cv2.imshow('vh_org', img_vh)

    img_vh = refine_vh12(img_vh, cols)
    if (not np.any(img_vh)) or len(img_vh) == 0:
        return []

    if debug_mode:
        cv2.imshow("vh", img_vh)
        cv2.waitKey(0)

    #----- cell index positions -------
    cols_sp, cols_ep = get_index_x(img_vh, int((cols[1] + cols[2]) / 2))
    if len(cols_sp) < 12 or len(cols_ep) < 12:
        return []
    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2
    h21, h22 = cols_sp[2] + delta, cols_ep[2] + delta2
    h61, h62, h71, h72 = cols_sp[4] + delta, cols_ep[4] + delta2, cols_sp[5] + delta, cols_ep[5] + delta2
    h91, h92 = cols_sp[11] + delta, cols_ep[11] + delta2

    t = int((h11 + h12) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 2 or len(ep) < 2:
        return []
    w11, w12 = sp[1], sp[1] + 700
    t = int((sp[0] + ep[0]) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 12 or len(cols_ep) < 12:
        return []
    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2

    t = int((h21 + h22) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 2 or len(ep) < 2:
        return []
    w21, w22 = sp[1], ep[1]
    t = int((sp[0] + ep[0]) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 12 or len(cols_ep) < 12:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h21, h22 = cols_sp[2] + delta, cols_ep[2] + delta2

    t = int((h61 + h62) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 5 or len(ep) < 5:
        return []
    w61, w62, w71, w72 = sp[1], ep[1], sp[1], ep[1]
    t = int((w61 + w62) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 12 or len(cols_ep) < 12:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h61, h62, h71, h72 = cols_sp[4]+delta, cols_ep[4]+delta2, cols_sp[5]+delta, cols_ep[5]+delta2

    t = int((h91 + h92) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 5 or len(ep) < 5:
        return []
    w91, w92 = sp[4], ep[4]
    t = int((w91 + w92) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 12 or len(cols_ep) < 12:
        cols_sp, cols_ep = get_index_x(img_vh, t - 10)
    h91, h92 = cols_sp[11] + delta, cols_ep[11] + delta2

    if OCR_IMG_TYPE:
        img_1 = img_table[h11:h12, w11:w12].copy()
        img_1 = bbox_noise_remove(img_1)
        dh11, dh12, dw11, dw12 = bbox(img_1)
        img_1 = img_1[dh11:dh12, dw11:dw12]

        img_2 = img_table[h21:h22, w21:w22].copy()
        img_2 = bbox_noise_remove(img_2)
        dh21, dh22, dw21, dw22 = bbox(img_2)
        img_2 = img_2[dh21:dh22, dw21:dw22]

        img_6 = img_table[h61:h62, w61:w62]
        img_6 = bbox_noise_remove(img_6)
        dh61, dh62, dw61, dw62 = bbox(img_6)
        img_6 = img_6[dh61:dh62, dw61:dw62]

        img_7 = img_table[h71:h72, w71:w72]
        img_7 = bbox_noise_remove(img_7)
        dh71, dh72, dw71, dw72 = bbox(img_7)
        img_7 = img_7[dh71:dh72, dw71:dw72]

        img_9 = img_table[h91:h92, w91:w92]
        img_9 = bbox_noise_remove(img_9)
        dh91, dh92, dw91, dw92 = bbox(img_9)
        img_9 = img_9[dh91:dh92, dw91:dw92+1]
        v = 0
    else:
        img_1 = img_table[h11:h12, w11:w12].copy()
        img_1 = bbox_noise_remove(img_1)
        dh11, dh12, dw11, dw12 = bbox(img_1)
        img_1 = img_smooth[h11 + dh11:h11 + dh12, w11 + dw11:w11 + dw12]

        img_2 = img_table[h21:h22, w21:w22].copy()
        img_2 = bbox_noise_remove(img_2)
        dh21, dh22, dw21, dw22 = bbox(img_2)
        img_2 = img_smooth[h21 + dh21:h21 + dh22, w21 + dw21:w21 + dw22]

        img_6 = img_table[h61:h62, w61:w62]
        img_6 = bbox_noise_remove(img_6)
        dh61, dh62, dw61, dw62 = bbox(img_6)
        img_6 = img_smooth[h61 + dh61:h61 + dh62, w61 + dw61:w61 + dw62]

        img_7 = img_table[h71:h72, w71:w72]
        img_7 = bbox_noise_remove(img_7)
        dh71, dh72, dw71, dw72 = bbox(img_7)
        img_7 = img_smooth[h71 + dh71:h71 + dh72, w71 + dw71:w71 + dw72]

        img_9 = img_table[h91:h92, w91:w92]
        img_9 = bbox_noise_remove(img_9)
        dh91, dh92, dw91, dw92 = bbox(img_9)
        img_9 = img_smooth[h91 + dh91:h91 + dh92, w91 + dw91:w91 + dw92]

        v = 255

    bs = 8

    img_1 = cv2.copyMakeBorder(img_1, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_2 = cv2.copyMakeBorder(img_2, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_6 = cv2.copyMakeBorder(img_6, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_7 = cv2.copyMakeBorder(img_7, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_9 = cv2.copyMakeBorder(img_9, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)

    if debug_mode:
        cv2.imwrite('table.png', img_table)
        cv2.imshow('1', img_1)
        cv2.imshow('2', img_2)
        cv2.imshow('6', img_6)
        cv2.imshow('7', img_7)
        cv2.imshow('9', img_9)
        cv2.waitKey(0)



    str1 = pytesseract.image_to_string(img_1, config=text_config, lang='eng')
    str2 = pytesseract.image_to_string(img_2, config=digit_config)
    str6 = pytesseract.image_to_string(img_6, config=digit_config)
    str7 = pytesseract.image_to_string(img_7, config=digit_config)
    str9 = pytesseract.image_to_string(img_9, config=digit_config)
    return [str1, str2, '', '', '', str6, str7, '', str9]
def refine_vh14_2(img_vh, cols):
    bd = 3
    dbd = 150
    delta = 2
    delta2 = 2

    # paint white value in border
    img_vh[:, 0:6] = 255
    img_vh[0:6, :] = 255

    cols_sp, cols_ep = get_index_x(img_vh, int((cols[1] + cols[2]) / 2))
    if len(cols_sp) < 14 or len(cols_ep) < 14:
        return []

    h01, h02 = cols_sp[0], cols_ep[0]
    t = int((h01 + h02) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    t = 0
    if len(sp) > 1 and sp[-1] > img_vh.shape[1] - 200:
        t = sp[-1]
    if t > 0:
        img_vh = img_vh[:, :t -3]


    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2
    h21, h22, h31, h32 = cols_sp[2] + delta, cols_ep[2] + delta2, cols_sp[2] + delta, cols_ep[2] + delta2
    h41, h42, h51, h52 = cols_sp[3] + delta, cols_ep[3] + delta2, cols_sp[3] + delta, cols_ep[3] + delta2
    h61, h62, h71, h72, h81, h82 = cols_sp[5] + delta, cols_ep[5] + delta2, cols_sp[6] + delta, cols_ep[6] + delta2, \
                                   cols_sp[7] + delta, cols_ep[7] + delta2
    h91, h92 = cols_sp[13] + delta, cols_ep[13] + delta2

    t = int((h11 + h12) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h11, h12, bd, dbd, sp, ep, [cols[0]])

    t = int((h21 + h22) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h21, h22, bd, dbd, sp, ep, [cols[0], cols[2]])

    t = int((h41 + h42) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h41, h42, bd, dbd, sp, ep, [cols[0], cols[2]])

    t = int((h61 + h62) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h61, h62, bd, dbd, sp, ep, [cols[1], cols[2]])

    t = int((h71 + h72) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h71, h72, bd, dbd, sp, ep, [cols[1], cols[2]])

    t = int((h81 + h82) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h81, h82, bd, dbd, sp, ep, [cols[1], cols[2]])

    t = int((h91 + h92) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h91, h92, bd, dbd, sp, ep, [cols[1], cols[2], cols[4]])

    return img_vh
def refine_vh14(img_vh, cols):
    bd = 3
    dbd = 150
    delta = 2
    delta2 = 2

    # paint white value in border
    img_vh[:, 0:6] = 255
    img_vh[0:6, :] = 255

    cols_sp, cols_ep = get_index_x(img_vh, int((cols[1] + cols[2]) / 2))
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        return []

    h01, h02 = cols_sp[0], cols_ep[0]
    t = int((h01 + h02) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    t = 0
    if len(sp) > 1 and sp[-1] > img_vh.shape[1] - 200:
        t = sp[-1]
    if t > 0:
        img_vh = img_vh[:, :t - 3]



    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2
    h21, h22, h31, h32 = cols_sp[2] + delta, cols_ep[2] + delta2, cols_sp[2] + delta, cols_ep[2] + delta2
    h41, h42, h51, h52 = cols_sp[3] + delta, cols_ep[3] + delta2, cols_sp[3] + delta, cols_ep[3] + delta2
    h61, h62, h71, h72, h81, h82 = cols_sp[5] + delta, cols_ep[5] + delta2, cols_sp[6] + delta, cols_ep[6] + delta2, \
                                   cols_sp[7] + delta, cols_ep[7] + delta2
    h91, h92 = cols_sp[14] + delta, cols_ep[14] + delta2
    h101, h102 = cols_sp[13] + delta, cols_ep[13] + delta2

    t = int((h11 + h12) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h11, h12, bd, dbd, sp, ep, [cols[0]])

    t = int((h21 + h22) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h21, h22, bd, dbd, sp, ep, [cols[0], cols[2]])

    t = int((h41 + h42) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h41, h42, bd, dbd, sp, ep, [cols[0], cols[2]])

    t = int((h61 + h62) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h61, h62, bd, dbd, sp, ep, [cols[1], cols[2]])

    t = int((h71 + h72) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h71, h72, bd, dbd, sp, ep, [cols[1], cols[2]])

    t = int((h81 + h82) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h81, h82, bd, dbd, sp, ep, [cols[1], cols[2]])

    t = int((h91 + h92) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h91, h92, bd, dbd, sp, ep, [cols[1], cols[2], cols[4]])

    t = int((h101 + h102) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h101, h102, bd, dbd, sp, ep, [cols[1], cols[2], cols[4]])

    return img_vh
def get_ncell_14_2(img_table_org):
    h, w = img_table_org.shape
    cols = np.array([0.2132, 0.3729, 0.489, 0.6817, 0.8774])
    cols = cols * w
    cols = cols.astype(np.int)

    dst_img_table = cv2.GaussianBlur(img_table_org, (5, 5), 0)
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    img_smooth = cv2.filter2D(dst_img_table, -1, kernel_sharpening)

    t, _ = cv2.threshold(img_table_org, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, img_cell = cv2.threshold(img_table_org, t + 30, 255, cv2.THRESH_BINARY_INV)
    t = max(30, t - THRESOCR_DEL)
    _, img_table = cv2.threshold(img_smooth, t, 255, cv2.THRESH_BINARY_INV)

    delta = 1
    delta2 = 0
    kernel_len_v = img_cell.shape[0] // 10
    kernel_len_h = img_cell.shape[1] // 10
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_v))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_h, 1))

    v_lines = cv2.erode(img_cell, ver_kernel, iterations=1)
    v_lines = cv2.dilate(v_lines, ver_kernel, iterations=1)

    h_lines = cv2.erode(img_cell, hor_kernel, iterations=1)
    h_lines = cv2.dilate(h_lines, hor_kernel, iterations=1)
    img_vh = cv2.addWeighted(v_lines, 1.0, h_lines, 1.0, 0.0)

    if debug_mode:
        cv2.imshow('vh_org', img_vh)

    img_vh = refine_vh14(img_vh, cols)
    if (not np.any(img_vh)) or len(img_vh) == 0:
        return []

    if debug_mode:
        cv2.imshow("vh", img_vh)
        cv2.waitKey(0)

    # ----- cell index positions -------
    cols_sp, cols_ep = get_index_x(img_vh, int((cols[1] + cols[2]) / 2))
    if len(cols_sp) < 14 or len(cols_ep) < 14:
        return []
    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2
    h21, h22, h31, h32 = cols_sp[2] + delta, cols_ep[2] + delta2, cols_sp[2] + delta, cols_ep[2] + delta2
    h41, h42, h51, h52 = cols_sp[3] + delta, cols_ep[3] + delta2, cols_sp[3] + delta, cols_ep[3] + delta2
    h61, h62, h71, h72, h81, h82 = cols_sp[5] + delta, cols_ep[5] + delta2, cols_sp[6] + delta, cols_ep[6] + delta2, \
                                   cols_sp[7] + delta, cols_ep[7] + delta2
    h91, h92 = cols_sp[13] + delta, cols_ep[13] + delta2

    t = int((h11 + h12) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 2 or len(ep) < 2:
        return []
    w11, w12 = sp[1], sp[1] + 700
    t = int((sp[0] + ep[0]) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 14 or len(cols_ep) < 14:
        return []
    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2

    t = int((h21 + h22) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 3 or len(ep) < 3:
        return []
    w21, w22, w31, w32 = sp[1], ep[1], sp[2], ep[2]
    w41, w42, w51, w52 = sp[1], ep[1], sp[2] + 300, ep[2]
    t = int((w21 + w22) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 14 or len(cols_ep) < 14:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h21, h22 = cols_sp[2] + delta, cols_ep[2] + delta2
    t = int((w31 + w32) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 14 or len(cols_ep) < 14:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h31, h32 = cols_sp[2] + delta, cols_ep[2] + delta2
    t = int((w41 + w42) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 14 or len(cols_ep) < 14:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h41, h42 = cols_sp[3] + delta, cols_ep[3] + delta2
    t = int((w51 + w52) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 14 or len(cols_ep) < 14:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h51, h52 = cols_sp[3] + delta, cols_ep[3] + delta2

    t = int((h61 + h62) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 3 or len(ep) < 3:
        return []
    w61, w62, w71, w72, w81, w82 = sp[1], ep[1], sp[1], ep[1], sp[1], ep[1]
    t = int((w61 + w62) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 14 or len(cols_ep) < 14:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h61, h62, h71, h72, h81, h82 = cols_sp[5] + delta, cols_ep[5] + delta2, cols_sp[6] + delta, cols_ep[6] + delta2, \
                                   cols_sp[7] + delta, cols_ep[7] + delta2

    t = int((h91 + h92) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 4 or len(ep) < 4:
        return []
    w91, w92 = sp[3], ep[3]
    t = int((w91 + w92) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 14 or len(cols_ep) < 14:
        cols_sp, cols_ep = get_index_x(img_vh, t - 10)
    h91, h92 = cols_sp[13] + delta, cols_ep[13] + delta2
    # -----

    cv2.imwrite('table.png', img_table)
    if OCR_IMG_TYPE:
        img_1 = img_table[h11:h12, w11:w12].copy()
        img_1 = bbox_noise_remove(img_1)
        dh11, dh12, dw11, dw12 = bbox(img_1)
        img_1 = img_1[dh11:dh12, dw11:dw12]

        img_2 = img_table[h21:h22, w21:w22].copy()
        img_2 = bbox_noise_remove(img_2)
        dh21, dh22, dw21, dw22 = bbox(img_2)
        img_2 = img_2[dh21:dh22, dw21:dw22]

        img_3 = img_table[h31:h32, w31:w32]
        img_3 = bbox_noise_remove(img_3)
        dh31, dh32, dw31, dw32 = bbox(img_3)
        img_3 = img_3[dh31:dh32, dw31:dw32]

        img_4 = img_table[h41:h42, w41:w42]
        img_4 = bbox_noise_remove(img_4)
        dh41, dh42, dw41, dw42 = bbox(img_4)
        img_4 = img_4[dh41:dh42, dw41:dw42]

        img_5 = img_table[h51:h52, w51:w52]
        img_5 = bbox_noise_remove(img_5)
        dh51, dh52, dw51, dw52 = bbox(img_5)
        img_5 = img_5[dh51:dh52, dw51:dw52]

        img_6 = img_table[h61:h62, w61:w62]
        img_6 = bbox_noise_remove(img_6)
        dh61, dh62, dw61, dw62 = bbox(img_6)
        img_6 = img_6[dh61:dh62, dw61:dw62]

        img_7 = img_table[h71:h72, w71:w72]
        img_7 = bbox_noise_remove(img_7)
        dh71, dh72, dw71, dw72 = bbox(img_7)
        img_7 = img_7[dh71:dh72, dw71:dw72]

        img_8 = img_table[h81:h82, w81:w82]
        img_8 = bbox_noise_remove(img_8)
        dh81, dh82, dw81, dw82 = bbox(img_8)
        img_8 = img_8[dh81:dh82, dw81:dw82]

        img_9 = img_table[h91:h92, w91:w92]
        img_9 = bbox_noise_remove(img_9)
        dh91, dh92, dw91, dw92 = bbox(img_9)
        img_9 = img_9[dh91:dh92, dw91:dw92 + 1]
        v = 0
    else:
        img_1 = img_table[h11:h12, w11:w12].copy()
        img_1 = bbox_noise_remove(img_1)
        dh11, dh12, dw11, dw12 = bbox(img_1)
        img_1 = img_smooth[h11 + dh11:h11 + dh12, w11 + dw11:w11 + dw12]

        img_2 = img_table[h21:h22, w21:w22].copy()
        img_2 = bbox_noise_remove(img_2)
        dh21, dh22, dw21, dw22 = bbox(img_2)
        img_2 = img_smooth[h21 + dh21:h21 + dh22, w21 + dw21:w21 + dw22]

        img_3 = img_table[h31:h32, w31:w32]
        img_3 = bbox_noise_remove(img_3)
        dh31, dh32, dw31, dw32 = bbox(img_3)
        img_3 = img_smooth[h31 + dh31:h31 + dh32, w31 + dw31:w31 + dw32]

        img_4 = img_table[h41:h42, w41:w42]
        img_4 = bbox_noise_remove(img_4)
        dh41, dh42, dw41, dw42 = bbox(img_4)
        img_4 = img_smooth[h41 + dh41:h41 + dh42, w41 + dw41:w41 + dw42]

        img_5 = img_table[h51:h52, w51:w52]
        img_5 = bbox_noise_remove(img_5)
        dh51, dh52, dw51, dw52 = bbox(img_5)
        img_5 = img_smooth[h51 + dh51:h51 + dh52, w51 + dw51:w51 + dw52]

        img_6 = img_table[h61:h62, w61:w62]
        img_6 = bbox_noise_remove(img_6)
        dh61, dh62, dw61, dw62 = bbox(img_6)
        img_6 = img_smooth[h61 + dh61:h61 + dh62, w61 + dw61:w61 + dw62]

        img_7 = img_table[h71:h72, w71:w72]
        img_7 = bbox_noise_remove(img_7)
        dh71, dh72, dw71, dw72 = bbox(img_7)
        img_7 = img_smooth[h71 + dh71:h71 + dh72, w71 + dw71:w71 + dw72]

        img_8 = img_table[h81:h82, w81:w82]
        img_8 = bbox_noise_remove(img_8)
        dh81, dh82, dw81, dw82 = bbox(img_8)
        img_8 = img_smooth[h81 + dh81:h81 + dh82, w81 + dw81:w81 + dw82]

        img_9 = img_table[h91:h92, w91:w92]
        img_9 = bbox_noise_remove(img_9)
        dh91, dh92, dw91, dw92 = bbox(img_9)
        img_9 = img_smooth[h91 + dh91:h91 + dh92, w91 + dw91:w91 + dw92]

        v = 255
    bs = 8
    img_1 = cv2.copyMakeBorder(img_1, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_2 = cv2.copyMakeBorder(img_2, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_3 = cv2.copyMakeBorder(img_3, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_4 = cv2.copyMakeBorder(img_4, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_5 = cv2.copyMakeBorder(img_5, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_6 = cv2.copyMakeBorder(img_6, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_7 = cv2.copyMakeBorder(img_7, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_8 = cv2.copyMakeBorder(img_8, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_9 = cv2.copyMakeBorder(img_9, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)

    if debug_mode:
        cv2.imshow('1', img_1)
        cv2.imshow('2', img_2)
        cv2.imshow('3', img_3)
        cv2.imshow('4', img_4)
        cv2.imshow('5', img_5)
        cv2.imshow('6', img_6)
        cv2.imshow('7', img_7)
        cv2.imshow('8', img_8)
        cv2.imshow('9', img_9)
        cv2.waitKey(0)

    str1 = pytesseract.image_to_string(img_1, config=text_config, lang='eng')
    str2 = pytesseract.image_to_string(img_2, config=digit_config)
    str3 = pytesseract.image_to_string(img_3, config=text_config, lang='eng')
    str4 = pytesseract.image_to_string(img_4, config=date_config)
    str5 = pytesseract.image_to_string(img_5, config=date_config)
    str6 = pytesseract.image_to_string(img_6, config=digit_config)
    str7 = pytesseract.image_to_string(img_7, config=digit_config)
    str8 = pytesseract.image_to_string(img_8, config=digit_config)
    str9 = pytesseract.image_to_string(img_9, config=digit_config)

    return [str1, str2, str3, str4, str5, str6, str7, str8, str9]
def get_ncell_14(img_table_org):
    h, w = img_table_org.shape
    cols = np.array([0.2132, 0.3729, 0.489, 0.6817, 0.8774])
    cols = cols * w
    cols = cols.astype(np.int)

    dst_img_table = cv2.GaussianBlur(img_table_org, (5, 5), 0)
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    img_smooth = cv2.filter2D(dst_img_table, -1, kernel_sharpening)

    t, _ = cv2.threshold(img_table_org, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, img_cell = cv2.threshold(img_table_org, t + 30, 255, cv2.THRESH_BINARY_INV)
    t = max(30, t - THRESOCR_DEL)
    _, img_table = cv2.threshold(img_smooth, t, 255, cv2.THRESH_BINARY_INV)

    delta = 1
    delta2 = 0
    kernel_len_v = img_cell.shape[0] // 10
    kernel_len_h = img_cell.shape[1] // 10
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_v))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_h, 1))

    v_lines = cv2.erode(img_cell, ver_kernel, iterations=1)
    v_lines = cv2.dilate(v_lines, ver_kernel, iterations=1)

    h_lines = cv2.erode(img_cell, hor_kernel, iterations=1)
    h_lines = cv2.dilate(h_lines, hor_kernel, iterations=1)
    img_vh = cv2.addWeighted(v_lines, 1.0, h_lines, 1.0, 0.0)

    if debug_mode:
        cv2.imshow('vh_org', img_vh)

    img_vh = refine_vh14(img_vh, cols)
    if (not np.any(img_vh)) or len(img_vh) == 0:
        return []

    if debug_mode:
        cv2.imshow("vh", img_vh)
        cv2.waitKey(0)

    # ----- cell index positions -------
    cols_sp, cols_ep = get_index_x(img_vh, int((cols[1] + cols[2]) / 2))
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        return []
    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2
    h21, h22, h31, h32 = cols_sp[2] + delta, cols_ep[2] + delta2, cols_sp[2] + delta, cols_ep[2] + delta2
    h41, h42, h51, h52 = cols_sp[3] + delta, cols_ep[3] + delta2, cols_sp[3] + delta, cols_ep[3] + delta2
    h61, h62, h71, h72, h81, h82 = cols_sp[5] + delta, cols_ep[5] + delta2, cols_sp[6] + delta, cols_ep[6] + delta2, \
                                   cols_sp[7] + delta, cols_ep[7] + delta2
    h91, h92 = cols_sp[14] + delta, cols_ep[14] + delta2
    h101, h102 = cols_sp[13] + delta, cols_ep[13] + delta2

    t = int((h11 + h12) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 2 or len(ep) < 2:
        return []
    w11, w12 = sp[1], sp[1] + 700
    t = int((sp[0] + ep[0]) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        return []
    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2

    t = int((h21 + h22) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 3 or len(ep) < 3:
        return []
    w21, w22, w31, w32 = sp[1], ep[1], sp[2], ep[2]
    w41, w42, w51, w52 = sp[1], ep[1], sp[2] + 300, ep[2]
    t = int((w21 + w22) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h21, h22 = cols_sp[2] + delta, cols_ep[2] + delta2
    t = int((w31 + w32) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h31, h32 = cols_sp[2] + delta, cols_ep[2] + delta2
    t = int((w41 + w42) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h41, h42 = cols_sp[3] + delta, cols_ep[3] + delta2
    t = int((w51 + w52) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h51, h52 = cols_sp[3] + delta, cols_ep[3] + delta2

    t = int((h61 + h62) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 3 or len(ep) < 3:
        return []
    w61, w62, w71, w72, w81, w82 = sp[1], ep[1], sp[1], ep[1], sp[1], ep[1]
    t = int((w61 + w62) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h61, h62, h71, h72, h81, h82 = cols_sp[5] + delta, cols_ep[5] + delta2, cols_sp[6] + delta, cols_ep[6] + delta2, \
                                   cols_sp[7] + delta, cols_ep[7] + delta2

    t = int((h91 + h92) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 4 or len(ep) < 4:
        return []
    w91, w92 = sp[3], ep[3]
    t = int((w91 + w92) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t - 10)
    h91, h92 = cols_sp[14] + delta, cols_ep[14] + delta2

    t = int((h101 + h102) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 4 or len(ep) < 4:
        return []
    w101, w102 = sp[3], ep[3]
    t = int((w101 + w102) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t - 10)
    h101, h102 = cols_sp[13] + delta, cols_ep[13] + delta2
    # -----

    cv2.imwrite('table.png', img_table)

    if OCR_IMG_TYPE:
        img_1 = img_table[h11:h12, w11:w12].copy()
        img_1 = bbox_noise_remove(img_1)
        dh11, dh12, dw11, dw12 = bbox(img_1)
        img_1 = img_1[dh11:dh12, dw11:dw12]

        img_2 = img_table[h21:h22, w21:w22].copy()
        img_2 = bbox_noise_remove(img_2)
        dh21, dh22, dw21, dw22 = bbox(img_2)
        img_2 = img_2[dh21:dh22, dw21:dw22]

        img_3 = img_table[h31:h32, w31:w32]
        img_3 = bbox_noise_remove(img_3)
        dh31, dh32, dw31, dw32 = bbox(img_3)
        img_3 = img_3[dh31:dh32, dw31:dw32]

        img_4 = img_table[h41:h42, w41:w42]
        img_4 = bbox_noise_remove(img_4)
        dh41, dh42, dw41, dw42 = bbox(img_4)
        img_4 = img_4[dh41:dh42, dw41:dw42]

        img_5 = img_table[h51:h52, w51:w52]
        img_5 = bbox_noise_remove(img_5)
        dh51, dh52, dw51, dw52 = bbox(img_5)
        img_5 = img_5[dh51:dh52, dw51:dw52]

        img_6 = img_table[h61:h62, w61:w62]
        img_6 = bbox_noise_remove(img_6)
        dh61, dh62, dw61, dw62 = bbox(img_6)
        img_6 = img_6[dh61:dh62, dw61:dw62]

        img_7 = img_table[h71:h72, w71:w72]
        img_7 = bbox_noise_remove(img_7)
        dh71, dh72, dw71, dw72 = bbox(img_7)
        img_7 = img_7[dh71:dh72, dw71:dw72]

        img_8 = img_table[h81:h82, w81:w82]
        img_8 = bbox_noise_remove(img_8)
        dh81, dh82, dw81, dw82 = bbox(img_8)
        img_8 = img_8[dh81:dh82, dw81:dw82]

        img_9 = img_table[h91:h92, w91:w92]
        img_9 = bbox_noise_remove(img_9)
        dh91, dh92, dw91, dw92 = bbox(img_9)
        img_9 = img_9[dh91:dh92, dw91:dw92 + 1]
        v = 0
    else:
        img_1 = img_table[h11:h12, w11:w12].copy()
        img_1 = bbox_noise_remove(img_1)
        dh11, dh12, dw11, dw12 = bbox(img_1)
        img_1 = img_smooth[h11 + dh11:h11 + dh12, w11 + dw11:w11 + dw12]

        img_2 = img_table[h21:h22, w21:w22].copy()
        img_2 = bbox_noise_remove(img_2)
        dh21, dh22, dw21, dw22 = bbox(img_2)
        img_2 = img_smooth[h21 + dh21:h21 + dh22, w21 + dw21:w21 + dw22]

        img_3 = img_table[h31:h32, w31:w32]
        img_3 = bbox_noise_remove(img_3)
        dh31, dh32, dw31, dw32 = bbox(img_3)
        img_3 = img_smooth[h31 + dh31:h31 + dh32, w31 + dw31:w31 + dw32]

        img_4 = img_table[h41:h42, w41:w42]
        img_4 = bbox_noise_remove(img_4)
        dh41, dh42, dw41, dw42 = bbox(img_4)
        img_4 = img_smooth[h41 + dh41:h41 + dh42, w41 + dw41:w41 + dw42]

        img_5 = img_table[h51:h52, w51:w52]
        img_5 = bbox_noise_remove(img_5)
        dh51, dh52, dw51, dw52 = bbox(img_5)
        img_5 = img_smooth[h51 + dh51:h51 + dh52, w51 + dw51:w51 + dw52]

        img_6 = img_table[h61:h62, w61:w62]
        img_6 = bbox_noise_remove(img_6)
        dh61, dh62, dw61, dw62 = bbox(img_6)
        img_6 = img_smooth[h61 + dh61:h61 + dh62, w61 + dw61:w61 + dw62]

        img_7 = img_table[h71:h72, w71:w72]
        img_7 = bbox_noise_remove(img_7)
        dh71, dh72, dw71, dw72 = bbox(img_7)
        img_7 = img_smooth[h71 + dh71:h71 + dh72, w71 + dw71:w71 + dw72]

        img_8 = img_table[h81:h82, w81:w82]
        img_8 = bbox_noise_remove(img_8)
        dh81, dh82, dw81, dw82 = bbox(img_8)
        img_8 = img_smooth[h81 + dh81:h81 + dh82, w81 + dw81:w81 + dw82]

        img_9 = img_table[h91:h92, w91:w92]
        img_9 = bbox_noise_remove(img_9)
        dh91, dh92, dw91, dw92 = bbox(img_9)
        img_9 = img_smooth[h91 + dh91:h91 + dh92, w91 + dw91:w91 + dw92]

        img_10 = img_table[h101:h102, w101:w102]
        img_10 = bbox_noise_remove(img_10)
        dh101, dh102, dw101, dw102 = bbox(img_10)
        img_10 = img_smooth[h101 + dh101:h101 + dh102, w101 + dw101:w101 + dw102]

        v = 255

    bs = 8
    img_1 = cv2.copyMakeBorder(img_1, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_2 = cv2.copyMakeBorder(img_2, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_3 = cv2.copyMakeBorder(img_3, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_4 = cv2.copyMakeBorder(img_4, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_5 = cv2.copyMakeBorder(img_5, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_6 = cv2.copyMakeBorder(img_6, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_7 = cv2.copyMakeBorder(img_7, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_8 = cv2.copyMakeBorder(img_8, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_9 = cv2.copyMakeBorder(img_9, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_10 = cv2.copyMakeBorder(img_10, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)

    if debug_mode:
        cv2.imshow('1', img_1)
        cv2.imshow('2', img_2)
        cv2.imshow('3', img_3)
        cv2.imshow('4', img_4)
        cv2.imshow('5', img_5)
        cv2.imshow('6', img_6)
        cv2.imshow('7', img_7)
        cv2.imshow('8', img_8)
        cv2.imshow('9', img_9)
        cv2.imshow('10', img_10)
        cv2.waitKey(0)

    str1 = pytesseract.image_to_string(img_1, config=text_config, lang='eng')
    str2 = pytesseract.image_to_string(img_2, config=digit_config)
    str3 = pytesseract.image_to_string(img_3, config=text_config, lang='eng')
    str4 = pytesseract.image_to_string(img_4, config=date_config)
    str5 = pytesseract.image_to_string(img_5, config=date_config)
    str6 = pytesseract.image_to_string(img_6, config=digit_config)
    str7 = pytesseract.image_to_string(img_7, config=digit_config)
    str8 = pytesseract.image_to_string(img_8, config=digit_config)
    str9 = pytesseract.image_to_string(img_9, config=digit_config)
    str10 = pytesseract.image_to_string(img_10, config=digit_config)

    if len(str9) > 0 and str9[-1] == '1':
        str9 = str9[:-1]

    # int string validate
    vstr = '0123456789., '
    flg = True
    for c in str9:
        if not c in vstr:
            flg = False
            break

    if not flg or len(str9) < 5:
       str9 = str10

    return [str1, str2, str3, str4, str5, str6, str7, str8, str9]
def refine_vh15(img_vh, cols):
    bd = 3
    dbd = 150
    delta = 2
    delta2 = 2

    # paint white value in border
    img_vh[:, 0:6] = 255
    img_vh[0:6, :] = 255

    cols_sp, cols_ep = get_index_x(img_vh, int((cols[1] + cols[2]) / 2))
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        return []

    h01, h02 = cols_sp[0], cols_ep[0]
    t = int((h01 + h02) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    t = 0
    if len(sp) > 1 and sp[-1] > img_vh.shape[1] - 200:
        t = sp[-1]
    if t > 0:
        img_vh = img_vh[:, :t - 3]



    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2
    h21, h22, h31, h32 = cols_sp[2] + delta, cols_ep[2] + delta2, cols_sp[2] + delta, cols_ep[2] + delta2
    h41, h42, h51, h52 = cols_sp[3] + delta, cols_ep[3] + delta2, cols_sp[3] + delta, cols_ep[3] + delta2
    h61, h62, h71, h72, h81, h82 = cols_sp[5] + delta, cols_ep[5] + delta2, cols_sp[6] + delta, cols_ep[6] + delta2, \
                                   cols_sp[7] + delta, cols_ep[7] + delta2
    h91, h92 = cols_sp[14] + delta, cols_ep[14] + delta2

    t = int((h11 + h12) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h11, h12, bd, dbd, sp, ep, [cols[0]])

    t = int((h21 + h22) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h21, h22, bd, dbd, sp, ep, [cols[0], cols[2]])

    t = int((h41 + h42) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h41, h42, bd, dbd, sp, ep, [cols[0], cols[2]])

    t = int((h61 + h62) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h61, h62, bd, dbd, sp, ep, [cols[1], cols[2]])

    t = int((h71 + h72) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h71, h72, bd, dbd, sp, ep, [cols[1], cols[2]])

    t = int((h81 + h82) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h81, h82, bd, dbd, sp, ep, [cols[1], cols[2]])

    t = int((h91 + h92) / 2)
    sp, ep = get_index_y(img_vh, t, min_dist=0)
    img_vh = refine_row(img_vh, h91, h92, bd, dbd, sp, ep, [cols[1], cols[2], cols[4]])

    return img_vh
def get_ncell_15(img_table_org):
    h, w = img_table_org.shape
    cols = np.array([0.2132, 0.3729, 0.489, 0.6817, 0.8774])
    cols = cols * w
    cols = cols.astype(np.int)

    # img_smooth = cv2.medianBlur(img_table_org, 1)
    # img_smooth = cv2.GaussianBlur(img_table_org, (3, 3), 0)
    dst_img_table = cv2.GaussianBlur(img_table_org, (5, 5), 0)
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    img_smooth = cv2.filter2D(dst_img_table, -1, kernel_sharpening)

    t, _ = cv2.threshold(img_table_org, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, img_cell = cv2.threshold(img_table_org, t + 30, 255, cv2.THRESH_BINARY_INV)
    t = max(30, t - THRESOCR_DEL)
    _, img_table = cv2.threshold(img_smooth, t, 255, cv2.THRESH_BINARY_INV)

    delta = 1
    delta2 = 0
    kernel_len_v = img_cell.shape[0] // 10
    kernel_len_h = img_cell.shape[1] // 10
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_v))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_h, 1))

    v_lines = cv2.erode(img_cell, ver_kernel, iterations=1)
    v_lines = cv2.dilate(v_lines, ver_kernel, iterations=1)

    h_lines = cv2.erode(img_cell, hor_kernel, iterations=1)
    h_lines = cv2.dilate(h_lines, hor_kernel, iterations=1)
    img_vh = cv2.addWeighted(v_lines, 1.0, h_lines, 1.0, 0.0)

    if debug_mode:
        cv2.imshow('vh_org', img_vh)

    img_vh = refine_vh15(img_vh, cols)
    if (not np.any(img_vh)) or len(img_vh) == 0:
        return []

    if debug_mode:
        cv2.imshow("vh", img_vh)
        cv2.waitKey(0)

    # ----- cell index positions -------
    cols_sp, cols_ep = get_index_x(img_vh, int((cols[1] + cols[2]) / 2))
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        return []
    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2
    h21, h22, h31, h32 = cols_sp[2] + delta, cols_ep[2] + delta2, cols_sp[2] + delta, cols_ep[2] + delta2
    h41, h42, h51, h52 = cols_sp[3] + delta, cols_ep[3] + delta2, cols_sp[3] + delta, cols_ep[3] + delta2
    h61, h62, h71, h72, h81, h82 = cols_sp[5] + delta, cols_ep[5] + delta2, cols_sp[6] + delta, cols_ep[6] + delta2, \
                                   cols_sp[7] + delta, cols_ep[7] + delta2
    h91, h92 = cols_sp[14] + delta, cols_ep[14] + delta2

    t = int((h11 + h12) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 2 or len(ep) < 2:
        return []
    w11, w12 = sp[1], sp[1] + 700
    t = int((sp[0] + ep[0]) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        return []
    h11, h12 = cols_sp[1] + delta, cols_ep[1] + delta2

    t = int((h21 + h22) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 3 or len(ep) < 3:
        return []
    w21, w22, w31, w32 = sp[1], ep[1], sp[2], ep[2]
    w41, w42, w51, w52 = sp[1], ep[1], sp[2] + 300, ep[2]
    t = int((w21 + w22) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h21, h22 = cols_sp[2] + delta, cols_ep[2] + delta2
    t = int((w31 + w32) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h31, h32 = cols_sp[2] + delta, cols_ep[2] + delta2
    t = int((w41 + w42) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h41, h42 = cols_sp[3] + delta, cols_ep[3] + delta2
    t = int((w51 + w52) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h51, h52 = cols_sp[3] + delta, cols_ep[3] + delta2

    t = int((h61 + h62) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 3 or len(ep) < 3:
        return []
    w61, w62, w71, w72, w81, w82 = sp[1], ep[1], sp[1], ep[1], sp[1], ep[1]
    t = int((w61 + w62) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t + 10)
    h61, h62, h71, h72, h81, h82 = cols_sp[5] + delta, cols_ep[5] + delta2, cols_sp[6] + delta, cols_ep[6] + delta2, \
                                   cols_sp[7] + delta, cols_ep[7] + delta2

    t = int((h91 + h92) / 2)
    sp, ep = get_index_y(img_vh, t)
    if len(sp) < 4 or len(ep) < 4:
        return []
    w91, w92 = sp[3], ep[3]
    t = int((w91 + w92) / 2)
    cols_sp, cols_ep = get_index_x(img_vh, t)
    if len(cols_sp) < 15 or len(cols_ep) < 15:
        cols_sp, cols_ep = get_index_x(img_vh, t - 10)
    h91, h92 = cols_sp[14] + delta, cols_ep[14] + delta2
    # -----

    cv2.imwrite('table.png', img_table)

    if OCR_IMG_TYPE:
        img_1 = img_table[h11:h12, w11:w12].copy()
        img_1 = bbox_noise_remove(img_1)
        dh11, dh12, dw11, dw12 = bbox(img_1)
        img_1 = img_1[dh11:dh12, dw11:dw12]

        img_2 = img_table[h21:h22, w21:w22].copy()
        img_2 = bbox_noise_remove(img_2)
        dh21, dh22, dw21, dw22 = bbox(img_2)
        img_2 = img_2[dh21:dh22, dw21:dw22]

        img_3 = img_table[h31:h32, w31:w32]
        img_3 = bbox_noise_remove(img_3)
        dh31, dh32, dw31, dw32 = bbox(img_3)
        img_3 = img_3[dh31:dh32, dw31:dw32]

        img_4 = img_table[h41:h42, w41:w42]
        img_4 = bbox_noise_remove(img_4)
        dh41, dh42, dw41, dw42 = bbox(img_4)
        img_4 = img_4[dh41:dh42, dw41:dw42]

        img_5 = img_table[h51:h52, w51:w52]
        img_5 = bbox_noise_remove(img_5)
        dh51, dh52, dw51, dw52 = bbox(img_5)
        img_5 = img_5[dh51:dh52, dw51:dw52]

        img_6 = img_table[h61:h62, w61:w62]
        img_6 = bbox_noise_remove(img_6)
        dh61, dh62, dw61, dw62 = bbox(img_6)
        img_6 = img_6[dh61:dh62, dw61:dw62]

        img_7 = img_table[h71:h72, w71:w72]
        img_7 = bbox_noise_remove(img_7)
        dh71, dh72, dw71, dw72 = bbox(img_7)
        img_7 = img_7[dh71:dh72, dw71:dw72]

        img_8 = img_table[h81:h82, w81:w82]
        img_8 = bbox_noise_remove(img_8)
        dh81, dh82, dw81, dw82 = bbox(img_8)
        img_8 = img_8[dh81:dh82, dw81:dw82]

        img_9 = img_table[h91:h92, w91:w92]
        img_9 = bbox_noise_remove(img_9)
        dh91, dh92, dw91, dw92 = bbox(img_9)
        img_9 = img_9[dh91:dh92, dw91:dw92 + 1]
        v = 0
    else:
        img_1 = img_table[h11:h12, w11:w12].copy()
        img_1 = bbox_noise_remove(img_1)
        dh11, dh12, dw11, dw12 = bbox(img_1)
        img_1 = img_smooth[h11 + dh11:h11 + dh12, w11 + dw11:w11 + dw12]

        img_2 = img_table[h21:h22, w21:w22].copy()
        img_2 = bbox_noise_remove(img_2)
        dh21, dh22, dw21, dw22 = bbox(img_2)
        img_2 = img_smooth[h21 + dh21:h21 + dh22, w21 + dw21:w21 + dw22]

        img_3 = img_table[h31:h32, w31:w32]
        img_3 = bbox_noise_remove(img_3)
        dh31, dh32, dw31, dw32 = bbox(img_3)
        img_3 = img_smooth[h31 + dh31:h31 + dh32, w31 + dw31:w31 + dw32]

        img_4 = img_table[h41:h42, w41:w42]
        img_4 = bbox_noise_remove(img_4)
        dh41, dh42, dw41, dw42 = bbox(img_4)
        img_4 = img_smooth[h41 + dh41:h41 + dh42, w41 + dw41:w41 + dw42]

        img_5 = img_table[h51:h52, w51:w52]
        img_5 = bbox_noise_remove(img_5)
        dh51, dh52, dw51, dw52 = bbox(img_5)
        img_5 = img_smooth[h51 + dh51:h51 + dh52, w51 + dw51:w51 + dw52]

        img_6 = img_table[h61:h62, w61:w62]
        img_6 = bbox_noise_remove(img_6)
        dh61, dh62, dw61, dw62 = bbox(img_6)
        img_6 = img_smooth[h61 + dh61:h61 + dh62, w61 + dw61:w61 + dw62]

        img_7 = img_table[h71:h72, w71:w72]
        img_7 = bbox_noise_remove(img_7)
        dh71, dh72, dw71, dw72 = bbox(img_7)
        img_7 = img_smooth[h71 + dh71:h71 + dh72, w71 + dw71:w71 + dw72]

        img_8 = img_table[h81:h82, w81:w82]
        img_8 = bbox_noise_remove(img_8)
        dh81, dh82, dw81, dw82 = bbox(img_8)
        img_8 = img_smooth[h81 + dh81:h81 + dh82, w81 + dw81:w81 + dw82]

        img_9 = img_table[h91:h92, w91:w92]
        img_9 = bbox_noise_remove(img_9)
        dh91, dh92, dw91, dw92 = bbox(img_9)
        img_9 = img_smooth[h91 + dh91:h91 + dh92, w91 + dw91:w91 + dw92]

        v = 255

    bs = 8
    img_1 = cv2.copyMakeBorder(img_1, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_2 = cv2.copyMakeBorder(img_2, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_3 = cv2.copyMakeBorder(img_3, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_4 = cv2.copyMakeBorder(img_4, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_5 = cv2.copyMakeBorder(img_5, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_6 = cv2.copyMakeBorder(img_6, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_7 = cv2.copyMakeBorder(img_7, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_8 = cv2.copyMakeBorder(img_8, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)
    img_9 = cv2.copyMakeBorder(img_9, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, value=v)

    if debug_mode:
        cv2.imshow('1', img_1)
        cv2.imshow('2', img_2)
        cv2.imshow('3', img_3)
        cv2.imshow('4', img_4)
        cv2.imshow('5', img_5)
        cv2.imshow('6', img_6)
        cv2.imshow('7', img_7)
        cv2.imshow('8', img_8)
        cv2.imshow('9', img_9)
        cv2.waitKey(0)

    str1 = pytesseract.image_to_string(img_1, config=text_config, lang='eng')
    str2 = pytesseract.image_to_string(img_2, config=digit_config)
    str3 = pytesseract.image_to_string(img_3, config=text_config, lang='eng')
    str4 = pytesseract.image_to_string(img_4, config=date_config)
    str5 = pytesseract.image_to_string(img_5, config=date_config)
    str6 = pytesseract.image_to_string(img_6, config=digit_config)
    str7 = pytesseract.image_to_string(img_7, config=digit_config)
    str8 = pytesseract.image_to_string(img_8, config=digit_config)
    str9 = pytesseract.image_to_string(img_9, config=digit_config)

    return [str1, str2, str3, str4, str5, str6, str7, str8, str9]

# input : warping result
def get_chk_img2(org_img, t, scale = 3):
    # cropping of original image -----------------------------------------------------------------
    ncut = int(50 / scale)
    oh, ow = org_img.shape
    h2, w2 = int(oh / scale), int(ow / scale)
    img = cv2.resize(org_img, (w2, h2), cv2.INTER_AREA)

    _, thres_img = cv2.threshold(img, t + 60, 255, cv2.THRESH_BINARY_INV)
    thres_img[:, 0:ncut] = 0
    thres_img[:, -ncut:] = 0
    thres_img[0:ncut, :] = 0
    thres_img[-ncut:, :] = 0

    h, w = thres_img.shape
    line_remove_img = cv2.medianBlur(thres_img, 7)
    #cv2.imshow('median', line_remove_img)
    kernel = np.ones((7, 7), np.uint8)
    kernel2 = np.ones((11, 11), np.uint8)
    line_remove_img = cv2.dilate(line_remove_img, kernel2, None, iterations=1)
    line_remove_img = cv2.erode(line_remove_img, kernel, None, iterations=1)

    #cv2.imshow('line_removed', line_remove_img)
    #cv2.waitKey(0)
    cnts = cv2.findContours(line_remove_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    max_w = 0
    for cnt in cnts:
        if cv2.contourArea(cnt) < w * 10:
            continue
        box = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype='int')
        box = perspective.order_points(box)
        # check if box is table/check points
        bw = math.sqrt(
            (box[1][0] - box[0][0]) * (box[1][0] - box[0][0]) + (box[1][1] - box[0][1]) * (box[1][1] - box[0][1]))
        bh = math.sqrt(
            (box[1][0] - box[2][0]) * (box[1][0] - box[2][0]) + (box[1][1] - box[2][1]) * (box[1][1] - box[2][1]))
        if bw < w * 2 / 3 or bh < bw / 4:
            continue
        if max_w > bw:
            continue

        max_w = bw
        first = box[0]
        second = box[1]
        third = box[2]
        fourth = box[3]

    if max_w == 0:
        return ''

    check_pt0 = first
    check_pt1 = second
    check_pt2 = third
    check_pt3 = fourth

    if debug_mode:
        cv2.imshow('thres_cnt', thres_img)
        cv2.line(img, (check_pt0[0], check_pt0[1]), (check_pt1[0], check_pt1[1]), 0, 2)
        cv2.line(img, (check_pt1[0], check_pt1[1]), (check_pt2[0], check_pt2[1]), 0, 2)
        cv2.line(img, (check_pt2[0], check_pt2[1]), (check_pt3[0], check_pt3[1]), 0, 2)
        cv2.line(img, (check_pt3[0], check_pt3[1]), (check_pt0[0], check_pt0[1]), 0, 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    # get original coordinates
    check_pt0, check_pt1, check_pt2, check_pt3 = check_pt0 * scale, check_pt1 * scale, check_pt2 * scale, check_pt3 * scale

    check_w = int(math.sqrt(
        (check_pt0[0] - check_pt1[0]) * (check_pt0[0] - check_pt1[0]) + (check_pt0[1] - check_pt1[1]) * (
                check_pt0[1] - check_pt1[1])))
    check_h = int(math.sqrt(
        (check_pt2[0] - check_pt1[0]) * (check_pt2[0] - check_pt1[0]) + (check_pt2[1] - check_pt1[1]) * (
                check_pt2[1] - check_pt1[1])))

    dst_pts = np.float32([[0, 0], [check_w, 0], [check_w, check_h]])
    warp_mat_check = cv2.getAffineTransform(np.float32([check_pt0, check_pt1, check_pt2]), dst_pts)

    dst_img_check = cv2.warpAffine(org_img, warp_mat_check, (check_w, check_h))
    w0, w1, h0, h1 = int(check_w * 0.76), int(check_w * 0.92), int(check_w * 0.033), int(check_w * 0.082)
    dst_img_check = dst_img_check[h0:h1, w0:w1]

    # parsing of cropped image roi-------------------------------------------------------
    if debug_mode:
        cv2.imshow('c', dst_img_check)
        cv2.waitKey(0)
    minH = 25
    minW = 100
    thres_delta = 45
    thres_remove = 0.02

    # thresholding
    dst_img_check = cv2.GaussianBlur(dst_img_check, (5, 5), 0)
    thres, _ = cv2.threshold(dst_img_check, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thres = max(10, thres - thres_delta)
    _, thres_img = cv2.threshold(dst_img_check, thres, 255, cv2.THRESH_BINARY_INV)

    # initial cropping
    cnts = cv2.findContours(thres_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sx = 9999
    ex = -1
    sy = -1
    ey = 9999

    boxes = []
    cmax_height = -1
    csec_height = -1
    cmax_y = -1
    csec_y = -1
    cmax_width = -1
    csec_width = -1
    for cnt in cnts:
        box = cv2.boundingRect(cnt)
        boxes.append(box)
        if cmax_height < box[3]:
            cmax_height = box[3]
            cmax_width = box[2]
            cmax_y = box[1]
        elif csec_height < box[3]:
            csec_height = box[3]
            csec_width = box[2]
            csec_y = box[1]

    badds_max = []
    badds_sec = []
    nadd_max = 0
    nadd_sec = 0
    for box in boxes:
        if box[3] < minH or box[3] < cmax_height - 5 or math.fabs(box[1] - cmax_y) > 5:
            badds_max.append(False)
        else:
            badds_max.append(True)
            nadd_max += 1
    for box in boxes:
        if box[3] < minH or box[3] < csec_height - 5 or math.fabs(box[1] - csec_y) > 5:
            badds_sec.append(False)
        else:
            badds_sec.append(True)
            nadd_sec += 1
    if nadd_max < 2 and nadd_sec < 2 and cmax_width < minW and csec_width < minW:
        print('check roi error')
        return ''
    if nadd_max < 2 and cmax_width < minW:
        badds_max = badds_sec

    icnt = 0
    for cnt in cnts:
        box = boxes[icnt]
        badd = badds_max[icnt]
        icnt += 1
        if not badd:
            cv2.drawContours(thres_img, [cnt], -1, 0, -1)
            continue
        x0, x1, y0, y1 = box[0], box[0] + box[2], box[1], box[1] + box[3]
        if (sx > x0): sx = x0
        if (sy < y0): sy = y0
        if (ex < x1): ex = x1
        if (ey > y1): ey = y1
    if sx == 9999 or sy == 9999 or ex == -1 or ey == -1:
        print('minH threshold error')
        return ''

    if sy >= ey:
        print('Second minH error')
        return ''

    res = thres_img[sy:ey, sx:ex].copy()
    if debug_mode:
        cv2.imshow('res', res)
        cv2.waitKey(0)

    # remove little connections
    cols = np.sum(res, axis=0)
    rows = np.sum(res, axis=1)
    col_thres = thres_remove * (ex - sx + 1) * 255
    row_thres = thres_remove * (ey - sy + 1) * 255
    for i in range(len(cols)):
        if cols[i] < col_thres: res[:, i] = 0
    for i in range(len(rows)):
        if rows[i] < row_thres: res[i, :] = 0

    # resize
    nh = 30
    bs = int(nh / 4)
    h, w = res.shape
    nw = int(nh/h*w)
    res = cv2.resize(res, (nw, nh), cv2.INTER_CUBIC)

    res2 = cv2.copyMakeBorder(res, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, 0)
    digit_config = r'--oem 3 --psm 7 outputbase digits'
    strcheck = pytesseract.image_to_string(res2, config=digit_config)
    if debug_mode:
        print ('check : ' + strcheck)
        cv2.imshow('res2', res2)
        cv2.waitKey(0)

    # check result string
    available_str = '0123456789 '
    flg = True
    for c in strcheck:
        if not c in available_str:
            flg = False
            break
    if len(strcheck) < 3:
        flg = False
    if not flg:
        return ''
    return strcheck

def get_chk_img1(org_img, t, scale = 3):
    # cropping of original image -----------------------------------------------------------------
    ncut = int(50 / scale)
    oh, ow = org_img.shape
    h2, w2 = int(oh / scale), int(ow / scale)
    img = cv2.resize(org_img, (w2, h2), cv2.INTER_AREA)

    _, thres_img = cv2.threshold(img, t + 60, 255, cv2.THRESH_BINARY_INV)
    thres_img[:, 0:ncut] = 0
    thres_img[:, -ncut:] = 0
    thres_img[0:ncut, :] = 0
    thres_img[-ncut:, :] = 0

    h, w = thres_img.shape
    line_remove_img = cv2.medianBlur(thres_img, 7)
    #cv2.imshow('med', line_remove_img)
    kernel = np.ones((11, 11), np.uint8)
    kernel2 = np.ones((17, 17), np.uint8)
    line_remove_img = cv2.dilate(line_remove_img, kernel2, None, iterations=1)
    line_remove_img = cv2.erode(line_remove_img, kernel, None, iterations=1)

    cnts = cv2.findContours(line_remove_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    max_w = 0
    for cnt in cnts:
        if cv2.contourArea(cnt) < w * 10:
            continue
        box = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype='int')
        box = perspective.order_points(box)
        # check if box is table/check points
        bw = math.sqrt(
            (box[1][0] - box[0][0]) * (box[1][0] - box[0][0]) + (box[1][1] - box[0][1]) * (box[1][1] - box[0][1]))
        bh = math.sqrt(
            (box[1][0] - box[2][0]) * (box[1][0] - box[2][0]) + (box[1][1] - box[2][1]) * (box[1][1] - box[2][1]))
        if bw < w * 2 / 3 or bh < bw / 4:
            continue
        if max_w > bw:
            continue

        max_w = bw
        first = box[0]
        second = box[1]
        third = box[2]
        fourth = box[3]

    if max_w == 0:
        return ''

    check_pt0 = first
    check_pt1 = second
    check_pt2 = third
    check_pt3 = fourth

    if debug_mode:
        cv2.imshow('thres_cnt', thres_img)
        cv2.line(img, (check_pt0[0], check_pt0[1]), (check_pt1[0], check_pt1[1]), 0, 2)
        cv2.line(img, (check_pt1[0], check_pt1[1]), (check_pt2[0], check_pt2[1]), 0, 2)
        cv2.line(img, (check_pt2[0], check_pt2[1]), (check_pt3[0], check_pt3[1]), 0, 2)
        cv2.line(img, (check_pt3[0], check_pt3[1]), (check_pt0[0], check_pt0[1]), 0, 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    # get original coordinates
    check_pt0, check_pt1, check_pt2, check_pt3 = check_pt0 * scale, check_pt1 * scale, check_pt2 * scale, check_pt3 * scale

    check_w = int(math.sqrt(
        (check_pt0[0] - check_pt1[0]) * (check_pt0[0] - check_pt1[0]) + (check_pt0[1] - check_pt1[1]) * (
                check_pt0[1] - check_pt1[1])))
    check_h = int(math.sqrt(
        (check_pt2[0] - check_pt1[0]) * (check_pt2[0] - check_pt1[0]) + (check_pt2[1] - check_pt1[1]) * (
                check_pt2[1] - check_pt1[1])))

    dst_pts = np.float32([[0, 0], [check_w, 0], [check_w, check_h]])
    warp_mat_check = cv2.getAffineTransform(np.float32([check_pt0, check_pt1, check_pt2]), dst_pts)

    dst_img_check = cv2.warpAffine(org_img, warp_mat_check, (check_w, check_h))
    w0, w1, h0, h1 = int(check_w * 0.73), int(check_w * 0.91), int(check_w * 0.024), int(check_w * 0.072)
    dst_img_check = dst_img_check[h0:h1, w0:w1]

    # parsing of cropped image roi-------------------------------------------------------
    if debug_mode:
        cv2.imshow('c', dst_img_check)
        cv2.waitKey(0)

    minH = 20
    minW = 100
    thres_delta = 45
    thres_remove = 0.02

    # thresholding
    dst_img_check = cv2.GaussianBlur(dst_img_check, (5, 5), 0)
    thres, _ = cv2.threshold(dst_img_check, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thres = max(10, thres - thres_delta)
    _, thres_img = cv2.threshold(dst_img_check, thres, 255, cv2.THRESH_BINARY_INV)

    #cv2.imshow('thres', thres_img)
    #cv2.waitKey(0)
    # initial cropping
    cnts = cv2.findContours(thres_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sx = 9999
    ex = -1
    sy = -1
    ey = 9999

    boxes = []
    cmax_height = -1
    csec_height = -1
    cmax_y = -1
    csec_y = -1
    cmax_width = -1
    csec_width = -1
    for cnt in cnts:
        box = cv2.boundingRect(cnt)
        boxes.append(box)
        if cmax_height < box[3]:
            cmax_height = box[3]
            cmax_width = box[2]
            cmax_y = box[1]
        elif csec_height < box[3]:
            csec_height = box[3]
            csec_width = box[2]
            csec_y = box[1]

    badds_max = []
    badds_sec = []
    nadd_max = 0
    nadd_sec = 0
    for box in boxes:
        if box[3] < minH or box[3] < cmax_height - 5 or math.fabs(box[1] - cmax_y) > 5:
            badds_max.append(False)
        else:
            badds_max.append(True)
            nadd_max += 1
    for box in boxes:
        if box[3] < minH or box[3] < csec_height - 5 or math.fabs(box[1] - csec_y) > 5:
            badds_sec.append(False)
        else:
            badds_sec.append(True)
            nadd_sec += 1
    if nadd_max < 2 and nadd_sec < 2 and cmax_width < minW and csec_width < minW:
        print ('check roi error')
        return ''
    if nadd_max < 2 and cmax_width < minW:
        badds_max = badds_sec

    icnt = 0
    for cnt in cnts:
        box = boxes[icnt]
        badd = badds_max[icnt]
        icnt += 1
        if not badd:
            cv2.drawContours(thres_img, [cnt], -1, 0, -1)
            continue
        x0, x1, y0, y1 = box[0], box[0] + box[2], box[1], box[1] + box[3]
        if (sx > x0): sx = x0
        if (sy < y0): sy = y0
        if (ex < x1): ex = x1
        if (ey > y1): ey = y1
    if sx == 9999 or sy == 9999 or ex == -1 or ey == -1:
        print ('minH threshold error')
        return ''

    if sy >= ey:
        print ('Second minH error')
        return ''
    res = thres_img[sy:ey, sx:ex].copy()
    if debug_mode:
        cv2.imshow('res', res)
        cv2.waitKey(0)


    # remove little connections
    cols = np.sum(res, axis=0)
    rows = np.sum(res, axis=1)
    col_thres = thres_remove * (ex - sx + 1) * 255
    row_thres = thres_remove * (ey - sy + 1) * 255
    for i in range(len(cols)):
        if cols[i] < col_thres: res[:, i] = 0
    for i in range(len(rows)):
        if rows[i] < row_thres: res[i, :] = 0

    # resize
    nh = 30
    bs = int(nh / 4)
    h, w = res.shape
    nw = int(nh / h * w)
    res = cv2.resize(res, (nw, nh), cv2.INTER_CUBIC)

    res2 = cv2.copyMakeBorder(res, bs, bs, bs, bs, cv2.BORDER_CONSTANT, None, 0)
    digit_config = r'--oem 3 --psm 7 outputbase digits'
    strcheck = pytesseract.image_to_string(res2, config=digit_config)
    if debug_mode:
        print('check : ' + strcheck)
        cv2.imshow('res2', res2)
        cv2.waitKey(0)

    # check result string
    available_str = '0123456789 '
    flg = True
    for c in strcheck:
        if not c in available_str:
            flg = False
            break
    if len(strcheck) < 4:
        flg = False
    if not flg:
        return ''
    return strcheck

def get_chk_data(org_img, t):
    chk_data = get_chk_img1(org_img, t)
    if chk_data == '':
        print ('second loop')
        chk_data = get_chk_img2(org_img, t)
    return chk_data

def get_tbl_data(org_img, t, scale = 1.0):
    ncut = int(50 / scale)

    oh, ow = org_img.shape
    h2, w2 = int(oh / scale), int(ow / scale)
    img = cv2.resize(org_img, (w2, h2), cv2.INTER_AREA)

    pt1, pt2, pt3, pt4 = get_tbl_info(img, t,  ncut)

    if len(pt1) == 0:
        return []

    # get original pts
    table_pt0, table_pt1, table_pt2, table_pt3 = pt1 * scale, pt2 * scale, pt3 * scale, pt4 * scale
    table_pt0 = table_pt0.astype(np.int)
    table_pt1 = table_pt1.astype(np.int)
    table_pt2 = table_pt2.astype(np.int)
    table_pt3 = table_pt3.astype(np.int)

    if debug_mode:
        cv2.line(img, (table_pt0[0], table_pt0[1]), (table_pt1[0], table_pt1[1]), 0, 2)
        cv2.line(img, (table_pt1[0], table_pt1[1]), (table_pt2[0], table_pt2[1]), 0, 2)
        cv2.line(img, (table_pt2[0], table_pt2[1]), (table_pt3[0], table_pt3[1]), 0, 2)
        cv2.line(img, (table_pt3[0], table_pt3[1]), (table_pt0[0], table_pt0[1]), 0, 2)

    # warping of Original image
    table_w = int(math.sqrt(
        (table_pt0[0] - table_pt1[0]) * (table_pt0[0] - table_pt1[0]) + (table_pt0[1] - table_pt1[1]) * (
                    table_pt0[1] - table_pt1[1])))
    table_h = int(math.sqrt(
        (table_pt2[0] - table_pt1[0]) * (table_pt2[0] - table_pt1[0]) + (table_pt2[1] - table_pt1[1]) * (
                    table_pt2[1] - table_pt1[1])))

    dst_pts = np.float32([[0, 0], [0, table_h], [table_w, table_h]])
    warp_mat_table = cv2.getAffineTransform(np.float32([table_pt0, table_pt3, table_pt2]), dst_pts)

    global TableType
    if TableType == 14:
        table_h2 = int(table_h * 15 / 14)
        dst_img_table_2 = cv2.warpAffine(org_img, warp_mat_table, (table_w, table_h2))
    dst_img_table = cv2.warpAffine(org_img, warp_mat_table, (table_w, table_h))

    # parse table image and get results

    if TableType == 12:
        res = get_ncell_12(dst_img_table)
    elif TableType == 14:
        res = get_ncell_14(dst_img_table_2)
    else:
        res = get_ncell_15(dst_img_table)

    return res

def get_table_check(org_img, scale = 5):
    oh, ow = org_img.shape
    h2, w2 = int(oh / scale), int(ow / scale)
    img = cv2.resize(org_img, (w2, h2), cv2.INTER_AREA)
    t, thres_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    tbl_data = get_tbl_data(org_img, t)
    #tbl_data = ['', '', '', '', '', '', '', '', '']
    if len(tbl_data) == 0:
        print('table detection fail')

    chk_data = get_chk_data(org_img, t)
    if len(chk_data) == 0:
        print ('check detection fail')

    return [chk_data, tbl_data]

def date_parse(s):
    if len(s) == 6:
        return s[0] + '/'+s[1] + '/' + s[2:]
    elif len(s) == 7:
        s2 = s[:2]
        s3 = s[1:3]
        s2 = atoi(s2)
        s3 = atoi(s3)
        if s2 > 31:
            return s[0] + '/' + s[1:3] + '/' + s[3:]
        elif s3 > 12:
            return s[0] + '/' + s[1:3] + '/' + s[3:]
        else:
            return s[0:2] + '/' + s[2] + '/' + s[3:]
    else:
        return s[0:2] + '/' + s[2:4] + '/' + s[4:]

def int_parse(s):
    s = '{}'.format(s)
    s = s.replace('.', '')
    if len(s) < 3:
        return s
    return s[:-2] + '.' + s[-2:]

def string_post_proc(scheck, s1, s2, s3, s4, s5, s6, s7, s8, s9):
    arr = s3.split(':')
    if len(arr) > 1: s3 = arr[1]
    #s4 = date_parse(s4)
    #s5 = date_parse(s5)
    s6 = int_parse(s6)
    s7 = int_parse(s7)
    s8 = int_parse(s8)
    s9 = int_parse(s9)

    print ('----' + scheck + '-----')
    print ('1-'+s1, '2-'+s2, '3-'+s3, '4-'+s4, '5-'+s5, '6-'+s6, '7-'+s7, '8-'+s8, '9-'+s9)
    return [s1, s2, s3, s4, s5, s6, s7, s8, s9]

def extract_data(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if not np.any(img):
        print ('file is not picture')
        return 'N', []
    res = get_table_check(img)
    if len(res) == 0:
        print ('can not find check or table')
        return 'F', []
    else:
        sc = res[0]
        res = res[1]
        if len(res) == 0:
            print ('cant find table')
            return sc, []
        s1, s2, s3, s4, s5, s6, s7, s8, s9 = res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]
        ss = string_post_proc(sc, s1, s2, s3, s4, s5, s6, s7, s8, s9)
        return sc, ss

def scan_directory(root_path):
    files = []
    entries = os.listdir(root_path)
    for entry in entries:
        s_path = os.path.join(root_path, entry)
        if os.path.isfile(s_path):
            files.append(s_path)
            continue

        s_entries = os.listdir(s_path)
        for s_entry in s_entries:
            ss_path = os.path.join(s_path, s_entry)
            if os.path.isfile(ss_path):
                files.append(ss_path)
                continue

            ss_entries = os.listdir(ss_path)
            for ss_entry in ss_entries:
                sss_path = os.path.join(ss_path, ss_entry)
                if os.path.isdir(sss_path):
                    continue
                files.append(sss_path)

    checks = []
    s1s = []
    s2s = []
    s3s = []
    s4s = []
    s5s = []
    s6s = []
    s7s = []
    s8s = []
    s9s = []
    path = []
    for i in trange(len(files)):
        file = files[i]
        print (file)
        try:
            chk, tbl = extract_data(file)
        except Exception:
            print ('programming exception occur')
            chk, tbl = 'F', []
        if chk == 'N':
            continue
        if chk == 'F':
            checks.append('---')
            path.append(file)
            s1s.append('')
            s2s.append('')
            s3s.append('')
            s4s.append('')
            s5s.append('')
            s6s.append('')
            s7s.append('')
            s8s.append('')
            s9s.append('')
            continue
        if len(chk) < 3:
            checks.append('Wrong Check')
        else:
            checks.append(chk)
        path.append(file)
        if len(tbl) == 0:
            s1s.append('')
            s2s.append('')
            s3s.append('')
            s4s.append('')
            s5s.append('')
            s6s.append('')
            s7s.append('')
            s8s.append('')
            s9s.append('')
        else:
            s1s.append(tbl[0])
            s2s.append(tbl[1])
            s3s.append(tbl[2])
            s4s.append(tbl[3])
            s5s.append(tbl[4])
            s6s.append(tbl[5])
            s7s.append(tbl[6])
            s8s.append(tbl[7])
            s9s.append(tbl[8])

    df = pd.DataFrame({'Check' : checks, 'Path' : path, '1':s1s, '2':s2s, '3':s3s, '4':s4s, '5':s5s, '6':s6s, '7':s7s, '8':s8s, '9':s9s})
    if not os.path.exists('result.csv'):
        df.to_csv("result.csv")
    else:
        t = datetime.datetime.now()
        t = t.strftime("%d/%m/%Y %H:%M:%S")
        t = t.replace('/', '')
        t = t.replace(':', '_')
        t = t.replace(' ', '_')
        df.to_csv("result_new_{}.csv".format(t))

if __name__ == "__main__":
    if debug_mode:
        chk, tbl = extract_data(img_path)
    else:
        #scan_directory('../imgs/')
        scan_directory('../imgs/')


