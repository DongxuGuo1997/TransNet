import pickle
import numpy as np
import copy


def get_ped_ids_pie(annotations, image_set="all") -> list:
    """
    Returns all pedestrian ids
    :return: A list of pedestrian ids
    """
    pids = []
    image_set_nums = {'train': ['set01', 'set02', 'set04'],
                      'val': ['set05', 'set06'],
                      'test': ['set03'],
                      'all': ['set01', 'set02', 'set03',
                              'set04', 'set05', 'set06']}
    set_ids = image_set_nums[image_set]
    for sid in set_ids:
        for vid in sorted(annotations[sid]):
            pids.extend(annotations[sid][vid]['ped_annotations'].keys())

    return pids


def get_ped_info_pie(annotations, image_set="all") -> dict:
    """
        Get pedestrians' information,i.e. frames,bbox,occlusion, actions(walking or not),cross behavior.
        :param: annotations: PIE annotations in dictionary form
                image_set : str: train,val.test set split of PIE
        :return: information of all pedestrians in one video
    """
    assert image_set in ['train', 'test', 'val', "all"], "Image set should be train, test or val"
    pids = get_ped_ids_pie(annotations, image_set)
    dataset = annotations
    ped_info = {}
    for idx in pids:
        sn, vn, _ = idx.split("_")
        sid = "set{:02d}".format(int(sn))
        vid = "video_{:04d}".format(int(vn))
        ped_info[idx] = {}
        ped_info[idx]["set_number"] = sid
        ped_info[idx]["video_number"] = vid
        ped_info[idx]['frames'] = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['frames'])
        ped_info[idx]['bbox'] = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['bbox'])
        ped_info[idx]['occlusion'] = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['occlusion'])
        ped_info[idx]['action'] = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['behavior']['action'])
        ped_info[idx]['cross'] = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['behavior']['cross'])

    return ped_info


def filter_None(x):
    # small help function to filter None in list
    if x is None:
        return False
    else:
        return True


def ped_info_clean_pie(annotations, image_set="all") -> dict:
    """
     Remove all frames has occlusion tag = 2 (fully occluded)
    :param: annotations: PIE annotations in dictionary form
            image_set : image_set : str: train,val.test set split of PIE
    :return: cleaned information of all pedestrians in given set
    """
    assert image_set in ['train', 'test', 'val', "all"], "Image set should be train, test or val"
    ped_info = get_ped_info_pie(annotations, image_set)
    ids = list(ped_info.keys())
    # remove all frames with occlusion tag=2
    for idx in ids:
        occ = np.array(ped_info[idx]['occlusion'])
        full_occ = np.flatnonzero(occ == 2)
        # set fully occluded frames to None
        for i in range(len(full_occ)):
            ped_info[idx]['frames'][full_occ[i]] = None
            ped_info[idx]['bbox'][full_occ[i]] = None
            ped_info[idx]['action'][full_occ[i]] = None
            ped_info[idx]['occlusion'][full_occ[i]] = None
            ped_info[idx]['cross'][full_occ[i]] = None
        # filter all None values
        ped_info[idx]['frames'] = list(filter(filter_None, ped_info[idx]['frames']))
        ped_info[idx]['bbox'] = list(filter(filter_None, ped_info[idx]['bbox']))
        ped_info[idx]['action'] = list(filter(filter_None, ped_info[idx]['action']))
        ped_info[idx]['occlusion'] = list(filter(filter_None, ped_info[idx]['occlusion']))
        ped_info[idx]['cross'] = list(filter(filter_None, ped_info[idx]['cross']))

    return ped_info


def add_trans_label_pie(dataset, verbose=False) -> None:
    """
    Add labels to show the time (number of frames)
    away from next action transition
    """
    all_wts = 0  # walking to standing
    all_stw = 0  # standing to walking
    ped_ids = list(dataset.keys())
    for idx in ped_ids:
        action = dataset[idx]['action']
        frames = dataset[idx]['frames']
        n_frames = len(frames)
        dataset[idx]['next_transition'] = []
        stw_time = []
        wts_time = []
        for j in range(len(action) - 1):
            if action[j] == 0 and action[j + 1] == 1:
                all_stw += 1
                stw_time.append(frames[j + 1])
            elif action[j] == 1 and action[j + 1] == 0:
                all_wts += 1
                wts_time.append(frames[j + 1])
        # merge
        trans_time_ped = np.array(sorted(stw_time + wts_time))
        # set transition tag
        for i in range(n_frames):
            t = frames[i]
            future_trans_ped = trans_time_ped[trans_time_ped >= t]
            if future_trans_ped.size > 0:
                next_trans_ped = future_trans_ped[0]
                dataset[idx]['next_transition'].append(next_trans_ped - t)
            else:
                dataset[idx]['next_transition'].append(None)
    if verbose:
        print('\n')
        print('----------------------------------------------------------------')
        print(f'Total number of standing to walking transitions (raw): {all_stw}')
        print(f'Total number of walking to standing transitions  (raw): {all_wts}')

    return None


def build_ped_dataset_pie(pie_anns_path, image_set="all", verbose=False) -> dict:
    """
    Build pedestrian dataset from PIE annotations
    """
    assert image_set in ['train', 'test', 'val', "all"], "Image set should be train, test or val"
    pie_anns = pickle.load(open(pie_anns_path, 'rb'))
    ped_dataset = ped_info_clean_pie(pie_anns, image_set)
    add_trans_label_pie(ped_dataset, verbose)

    return ped_dataset


# -----------------------------------------------
class PieTransDataset:
    # dataset class for pedestrian samples
    def __init__(self, pie_anns_path, image_set="all", verbose=False):
        assert image_set in ['train', 'test', 'val', "all"]
        self.dataset = build_ped_dataset_pie(pie_anns_path, image_set=image_set, verbose=verbose)
        self.name = image_set

    def extract_trans_frame(self, mode="GO", verbose=False):
        dataset = self.dataset
        assert mode in ["GO", "STOP"], "Transition type should be STOP or GO"
        ids = list(dataset.keys())
        samples = {}
        j = 0
        for idx in ids:
            sid = copy.deepcopy(dataset[idx]['set_number'])
            vid = copy.deepcopy(dataset[idx]['video_number'])
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
                        key = "PG_" + new_id
                        old_id = f'{idx}/' + '{:03d}'.format(frames[i])
                if mode == "STOP":
                    if next_transition[i] == 0 and action[i] == 0 and action[i - d1] == 1 and action[i + d2] == 0:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "PS_" + new_id
                        old_id = f'{idx}/' + '{:03d}'.format(frames[i])
                if key is not None:
                    samples[key] = {}
                    samples[key]["source"] = "PIE"
                    samples[key]['set_number'] = sid
                    samples[key]['video_number'] = vid
                    samples[key]["old_id"] = old_id
                    samples[key]['frame'] = frames[i]
                    samples[key]['bbox'] = bbox[i]
                    samples[key]['action'] = action[i]
                    samples[key]['cross'] = cross[i]
        if verbose:
            print(f"Extract {len(samples.keys())} {mode} sample frames from PIE {self.name} set")

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
            sid = copy.deepcopy(dataset[idx]['set_number'])
            vid = copy.deepcopy(dataset[idx]['video_number'])
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
                        key = "PG_" + new_id
                        old_id = f'{idx}/' + '{:03d}'.format(frames[i])
                if mode == "STOP":
                    if next_transition[i] == 0 and action[i] == 0 and action[i - d1] == 1 and action[i + d2] == 0:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "PS_" + new_id
                        old_id = f'{idx}/' + '{:03d}'.format(frames[i])

                if key is not None:
                    samples[key] = {}
                    samples[key]["source"] = "PIE"
                    samples[key]["old_id"] = old_id
                    samples[key]['set_number'] = sid
                    samples[key]['video_number'] = vid
                    samples[key]['frame'] = frames[i::-step]
                    samples[key]['frame'].reverse()
                    samples[key]['bbox'] = bbox[i::-step]
                    samples[key]['bbox'].reverse()
                    samples[key]['action'] = action[i::-step]
                    samples[key]['action'].reverse()
                    samples[key]['cross'] = cross[i::-step]
                    samples[key]['cross'].reverse()

        return samples
