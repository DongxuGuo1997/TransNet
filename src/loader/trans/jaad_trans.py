import os
import numpy as np
import pickle
import copy


# --------------------------------------------------------------------
def get_split_vids(split_vids_path, image_set, subset='default') -> list:
    """
        Returns a list of video ids for a given data split
        :param:  split_vids_path: path of JAAD split
                image_set: Data split, train, test, val
                subset: "all", "default" or "high_resolution"
        :return: The list of video ids
        """
    assert image_set in ["train", "test", "val", "all"]
    vid_ids = []
    sets = [image_set] if image_set != 'all' else ['train', 'test', 'val']
    for s in sets:
        vid_id_file = os.path.join(split_vids_path, subset, s + '.txt')
        with open(vid_id_file, 'rt') as fid:
            vid_ids.extend([x.strip() for x in fid.readlines()])

    return vid_ids


def get_pedb_ids_jaad(annotations, vid):
    """
    Get pedestrians'(with behavior tags) ids in specific video.
    :param: dataset: JAAD raw data in dictionary form
            vid : video id (str)
    :return: pedestrians' ids

    """
    pedb_ids = []
    ped_keys = list(annotations[vid]['ped_annotations'].keys())
    for key in ped_keys:
        if 'b' in key:
            pedb_ids.append(key)

    return pedb_ids


def get_pedb_info_jaad(annotations, vid):
    """
    Get pedb information,i.e. frames,bbox,occlusion, actions(walking or not),cross behavior.
    :param: annotations: JAAD annotations in dictionary form
            vid : single video id (str)
    :return: information of all pedestrians in one video
    """
    ids = get_pedb_ids_jaad(annotations, vid)
    dataset = annotations
    pedb_info = {}
    for idx in ids:
        pedb_info[idx] = {}
        pedb_info[idx]['frames'] = []
        pedb_info[idx]['bbox'] = []
        pedb_info[idx]['occlusion'] = []
        pedb_info[idx]['action'] = []
        pedb_info[idx]['cross'] = []
        frames = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['frames'])
        bbox = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['bbox'])
        occlusion = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['occlusion'])
        action = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['behavior']['action'])
        cross = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['behavior']['cross'])
        for i in range(len(frames)):
            if action[i] in [0, 1]:  # sanity check if behavior label exists
                pedb_info[idx]['action'].append(action[i])
                pedb_info[idx]['frames'].append(frames[i])
                pedb_info[idx]['bbox'].append(bbox[i])
                pedb_info[idx]['occlusion'].append(occlusion[i])
                pedb_info[idx]['cross'].append(cross[i])

    return pedb_info


def filter_None(x):
    # Small help function to filter None in list
    if x is None:
        return False
    else:
        return True


def pedb_info_clean_jaad(annotations, vid) -> dict:
    """
     Remove all frames has occlusion tag = 2 (fully occluded)
         Get pedb information,i.e. frames,bbox,occlusion, actions(walking or not),cross behavior.
    :param: annotations: JAAD annotations in dictionary form
            vid : single video id (str)
    :return: cleaned information of all pedestrians in one video
    """
    pedb_info = get_pedb_info_jaad(annotations, vid)
    pids = list(pedb_info.keys())
    # remove all frames with occlusion tag=2
    for idx in pids:
        occ = np.array(pedb_info[idx]['occlusion'])
        full_occ = np.flatnonzero(occ == 2)
        # set fully occluded frames to None
        for i in range(len(full_occ)):
            pedb_info[idx]['frames'][full_occ[i]] = None
            pedb_info[idx]['bbox'][full_occ[i]] = None
            pedb_info[idx]['action'][full_occ[i]] = None
            pedb_info[idx]['occlusion'][full_occ[i]] = None
            pedb_info[idx]['cross'][full_occ[i]] = None
        # filter all None values
        pedb_info[idx]['frames'] = list(filter(filter_None, pedb_info[idx]['frames']))
        pedb_info[idx]['bbox'] = list(filter(filter_None, pedb_info[idx]['bbox']))
        pedb_info[idx]['action'] = list(filter(filter_None, pedb_info[idx]['action']))
        pedb_info[idx]['occlusion'] = list(filter(filter_None, pedb_info[idx]['occlusion']))
        pedb_info[idx]['cross'] = list(filter(filter_None, pedb_info[idx]['cross']))

    return pedb_info


def add_trans_label_jaad(dataset, verbose=False) -> None:
    """
    Add stop & go transition labels for every frame
    """
    all_wts = 0  # walking to standing(Stop)
    all_stw = 0  # standing to walking (Go)
    pids = list(dataset.keys())
    for idx in pids:
        action = dataset[idx]['action']
        frames = dataset[idx]['frames']
        n_frames = len(frames)
        dataset[idx]['next_transition'] = []
        stw_time = []
        wts_time = []

        for j in range(len(action) - 1):
            # stop and go transition
            if action[j] == 0 and action[j + 1] == 1:
                all_stw += 1
                stw_time.append(frames[j + 1])
            elif action[j] == 1 and action[j + 1] == 0:
                all_wts += 1
                wts_time.append(frames[j + 1])

        # merge
        trans_time = np.array(sorted(stw_time + wts_time))
        # set transition tag
        for i in range(n_frames):
            t_frame = frames[i]
            future_trans = trans_time[trans_time >= t_frame]
            if future_trans.size > 0:
                next_trans = future_trans[0]
                dataset[idx]['next_transition'].append(next_trans - t_frame)
            else:
                dataset[idx]['next_transition'].append(None)

    if verbose:
        print('\n')
        print('----------------------------------------------------------------')
        print(f'Total number of standing to walking transitions(raw): {all_stw}')
        print(f'Total number of walking to standing transitions(raw): {all_wts}')

    return None


def build_pedb_dataset_jaad(jaad_anns_path, split_vids_path, image_set="all", verbose=False) -> dict:
    """
    Build pedestrian dataset from jaad annotations
    """
    jaad_anns = pickle.load(open(jaad_anns_path, 'rb'))
    pedb_dataset = {}
    vids = get_split_vids(split_vids_path, image_set, subset='default')
    for vid in vids:
        pedb_info = pedb_info_clean_jaad(jaad_anns, vid)
        pids = list(pedb_info.keys())
        for idx in pids:
            pedb_dataset[idx] = {}
            pedb_dataset[idx]['video_number'] = vid
            pedb_dataset[idx]['frames'] = pedb_info[idx]['frames']
            pedb_dataset[idx]['bbox'] = pedb_info[idx]['bbox']
            pedb_dataset[idx]['action'] = pedb_info[idx]['action']
            pedb_dataset[idx]['occlusion'] = pedb_info[idx]['occlusion']
            pedb_dataset[idx]["cross"] = pedb_info[idx]["cross"]
    add_trans_label_jaad(pedb_dataset, verbose)

    return pedb_dataset


class JaadTransDataset:
    """
     dataset class for transition-related pedestrian samples in JAAD
    """

    def __init__(self, jaad_anns_path, split_vids_path, image_set="all", verbose=False):
        assert image_set in ['train', 'test', 'val', "all"], " Name should be train, test, val or all"
        self.dataset = build_pedb_dataset_jaad(jaad_anns_path, split_vids_path, image_set, verbose)
        self.name = image_set

    def extract_trans_frame(self, mode="GO", verbose=False):
        dataset = self.dataset
        assert mode in ["GO", "STOP"], "Transition type should be STOP or GO"
        ids = list(dataset.keys())
        samples = {}
        j = 0
        for idx in ids:
            vid_id = copy.deepcopy(dataset[idx]['video_number'])
            frames = copy.deepcopy(dataset[idx]['frames'])
            bbox = copy.deepcopy(dataset[idx]['bbox'])
            action = copy.deepcopy(dataset[idx]['action'])
            cross = copy.deepcopy(dataset[idx]['cross'])
            next_transition = copy.deepcopy(dataset[idx]["next_transition"])
            for i in range(len(frames)):
                key = None
                old_id = None
                d1 = min(i, 5)
                d2 = min(len(frames) - i - 1, 5)
                if mode == "GO":
                    if next_transition[i] == 0 and action[i] == 1 and action[i - d1] == 0 and action[i + d2] == 1:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "JG_" + new_id
                        old_id = f'{idx}/{vid_id}/' + '{:03d}'.format(frames[i])
                if mode == "STOP":
                    if next_transition[i] == 0 and action[i] == 0 and action[i - d1] == 1 and action[i + d2] == 0:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "JS_" + new_id
                        old_id = f'{idx}/{vid_id}/' + '{:03d}'.format(frames[i])
                if key is not None:
                    samples[key] = {}
                    samples[key]["source"] = "JAAD"
                    samples[key]["old_id"] = old_id
                    samples[key]['video_number'] = vid_id
                    samples[key]['frame'] = frames[i]
                    samples[key]['bbox'] = bbox[i]
                    samples[key]['action'] = action[i]
                    samples[key]['cross'] = cross[i]
        if verbose:
            print(f"Extract {len(samples.keys())} {mode} sample frames from JAAD {self.name} set")

        return samples

    def extract_trans_history(self, mode="GO", fps=30):
        dataset = self.dataset
        assert mode in ["GO", "STOP"], "Transition type should be STOP or GO"
        ids = list(dataset.keys())
        samples = {}
        j = 0
        step = 30 // fps
        assert isinstance(step, int)
        for idx in ids:
            vid_id = copy.deepcopy(dataset[idx]['video_number'])
            frames = copy.deepcopy(dataset[idx]['frames'])
            bbox = copy.deepcopy(dataset[idx]['bbox'])
            action = copy.deepcopy(dataset[idx]['action'])
            cross = copy.deepcopy(dataset[idx]['cross'])
            next_transition = copy.deepcopy(dataset[idx]["next_transition"])
            for i in range(len(frames)):
                key = None
                old_id = None
                d1 = min(i, 5)
                d2 = min(len(frames) - i - 1, 5)
                if mode == "GO":
                    if next_transition[i] == 0 and action[i] == 1 and action[i - d1] == 0 and action[i + d2] == 1:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "JG_" + new_id
                        old_id = f'{idx}/{vid_id}/' + '{:03d}'.format(frames[i])
                if mode == "STOP":
                    if next_transition[i] == 0 and action[i] == 0 and action[i - d1] == 1 and action[i + d2] == 0:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "JS_" + new_id
                        old_id = f'{idx}/{vid_id}/' + '{:03d}'.format(frames[i])
                if key is not None:
                    samples[key] = {}
                    samples[key]["source"] = "JAAD"
                    samples[key]["old_id"] = old_id
                    samples[key]['video_number'] = vid_id
                    samples[key]['frame'] = frames[i::-step]
                    samples[key]['frame'].reverse()
                    samples[key]['bbox'] = bbox[i::-step]
                    samples[key]['bbox'].reverse()
                    samples[key]['action'] = action[i::-step]
                    samples[key]['action'].reverse()
                    samples[key]['cross'] = cross[i::-step]
                    samples[key]['cross'].reverse()

        return samples
