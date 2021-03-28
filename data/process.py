import os
import cv2
import json
import pickle as pkl
from tqdm import tqdm


def gen_images(dbname):
    for split in ['train', 'test', 'val']:
        #if dbname == 'kinetics100':
        with open('splits/%s/%s.list'%(dbname, split), 'r') as f:
            vlist = [l.strip() for l in f.readlines()]

        vid2dict = dict()
        for v in vlist:
            clss, vid = v.split('/')[0], v.split('/')[1]
            vid2dict[vid] = clss
            if not os.path.exists('images/%s/%s'%(dbname, clss)):
                os.makedirs('images/%s/%s'%(dbname, clss))
        
        vdir = 'videos/%s/%s/'%(dbname, split)
        vnum = 0
        for video in tqdm(os.listdir(vdir)):
            vnum += 1
            if vnum < 5600:
                continue
            vid = video.split('.')[0]
            cap = cv2.VideoCapture(vdir+video)
            fnum = 0
            imdir = 'images/%s/%s/%s'%(dbname, vid2dict[vid], vid)
            if not os.path.exists(imdir):
                os.makedirs(imdir)
            while True:
                success, frame = cap.read()
                if not success:
                    break
                impath = 'images/%s/%s/%s/%s.jpg'%(dbname, vid2dict[vid], vid, fnum)

                cv2.imwrite(impath, frame)
                fnum += 1


def stat_kinetics_skeleton():
    miss_dict = {'train': [], 'test': [], 'val': []}
    jpath_dict = {'train':{}, 'test':{}, 'val':{}}
    for split in ['val', 'train', 'test']:
        rootdir = 'multi_modal/humanpose/kinetics100/'
        if not os.path.exists(rootdir+split):
            os.makedirs(rootdir+split)

        with open('splits/kinetics100/%s.list'%split) as f:
            flist = [l.strip() for l in f.readlines()]
        
        for fname in tqdm(flist):
            vid = fname.split("/")[1]
            json_file = vid[:11]+'.json'

            _skt_split = ''
            for skt_split in ['train', 'val']:
                if json_file not in os.listdir('multi_modal/humanpose/kinetics-skeleton/kinetics_%s'%skt_split):
                    continue
                else:
                    _skt_split = skt_split
                    jpath_dict[split][vid[:11]] = 'multi_modal/humanpose/kinetics-skeleton/kinetics_%s/%s'%(_skt_split, json_file)

            if _skt_split == '':
                miss_dict[split].append(vid.split('.')[0])
                print(len(miss_dict[split]), vid.split('.')[0])
                continue

            command = 'cp multi_modal/humanpose/kinetics-skeleton/kinetics_%s/%s %s/%s'%(_skt_split, json_file, rootdir, split)
            os.system(command)
            #print(command)
            #assert 1==0
            #break
    
    with open('multi_modal/humanpose/kinetics100/miss_dict.json', 'w') as f:
        json.dump(miss_dict, f)


def get_ssv2_100_videos():
    for split in ['train', 'test', 'val']:
        save_dir = 'videos/ssv2_100/%s'%split
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open('splits/ssv2/%s.list'%split, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        
        for line in tqdm(lines):
            clss = line.split('/')[0]
            vid = line.split('/')[1]
            command = 'cp ../../data/somethingsth/20bn-something-something-v2/%s.webm videos/ssv2_100/%s'%(vid, split)
            os.system(command)


def get_skt_ssv2_100():
    with open('multi_modal/humanpose/something-else/bounding_box_smthsmth_part4.json', 'r') as f:
        ss_data = json.load(f)
    print(type(ss_data))


def check_kinetics_skt():
    with open('multi_modal/humanpose/kinetics100/train/BwBms2Pht60.json', 'r') as f:
        data = json.load(f)
    print(type(data))
    print(list(data.keys()))
    print(data['data'][0])


def get_class_split():
    dataset = 'kinetics100'
    for split in ['train', 'val', 'test']:
        with open('splits/%s/%s.list'%(dataset, split), 'r') as f:
            videos = [v.strip() for v in f.readlines()]
        classes = []
        with open('splits/%s/%s_class.txt'%(dataset, split), 'w') as f:
            for v in videos:
                clss = v.split('/')[0]
                if clss not in classes:
                    f.writelines(clss+'\n')
                classes.append(clss)


if __name__ == '__main__':
    #check_kinetics_skt()
    #get_skt_ssv2_100()
    #stat_kinetics_skeleton()
    #gen_images('kinetics100')
    #get_ssv2_100_videos()
    get_class_split()
