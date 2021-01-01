import pandas as pd
import os
import numpy as np
import copy


def get_split_vids_titan(split_vids_path, image_set="all") -> list:
    """
        Returns a list of video ids for a given data split
        :param:  split_vids_path: path of TITAN split
                image_set: Data split, train, test, val
        :return: The list of video ids
        """
    assert image_set in ["train", "test", "val", "all"]
    vid_ids = []
    sets = [image_set + '_set'] if image_set != 'all' else ['train_set', 'test_set', 'val_set']
    for s in sets:
        vid_id_file = os.path.join(split_vids_path, s + '.txt')
        with open(vid_id_file, 'rt') as fid:
            vid_ids.extend([x.strip() for x in fid.readlines()])

    return vid_ids


def read_csv_titan(anns_dir, vid):
    video_number = int(vid.split("_")[1])
    df = pd.read_csv(os.path.join(anns_dir, vid + '.csv'))
    veh_rows = df[df['label'] != "person"].index
    df.drop(veh_rows, inplace=True)
    df.drop([df.columns[1], df.columns[7], df.columns[8], df.columns[9], df.columns[10],
             df.columns[11], df.columns[14], df.columns[15]],
            axis='columns', inplace=True)
    df.sort_values(by=['obj_track_id', 'frames'], inplace=True)
    ped_info_raw = df.values.tolist()
    pids = df['obj_track_id'].values.tolist()
    pids = list(set(list(map(int, pids))))

    return video_number, ped_info_raw, pids


def get_ped_info_titan(anns_dir, vids) -> dict:
    ped_info = {}
    for vid in vids:
        video_number, ped_info_raw, pids = read_csv_titan(anns_dir, vid)
        n = len(pids)
        ped_info[vid] = {}
        flag = 0
        for i in range(n):
            idx = f"ped_{video_number}_{i + 1}"
            ped_info[vid][idx] = {}
            ped_info[vid][idx]["frames"] = []
            ped_info[vid][idx]["bbox"] = []
            ped_info[vid][idx]["action"] = []
            # anns[vid][idx]["cross"] = []
            for j in range(flag, len(ped_info_raw)):
                if ped_info_raw[j][1] == pids[i]:
                    ele = ped_info_raw[j]
                    t = int(ele[0].split('.')[0])
                    # box = list([ele[3], ele[2], ele[3] + ele[5], ele[2] + ele[4]])
                    box = list(map(round, [ele[3], ele[2], ele[3] + ele[5], ele[2] + ele[4]]))
                    box = list(map(float, box))
                    action = 1 if ele[6] == "walking" else 0
                    ped_info[vid][idx]['frames'].append(t)
                    ped_info[vid][idx]['bbox'].append(box)
                    ped_info[vid][idx]['action'].append(action)
                else:
                    flag += len(ped_info[vid][idx]["frames"])
                    break
            ped_info[vid][idx]['old_id'] = vid + f'_{pids[i]}'

    return ped_info


def convert_anns_titan(anns_dir, vids) -> dict:
    anns = {}
    for vid in vids:
        video_number, ped_info_raw, pids = read_csv_titan(anns_dir, vid)
        n = len(pids)
        flag = 0
        for i in range(n):
            idx = f"ped_{video_number}_{i + 1}"
            anns[idx] = {}
            anns[idx]["frames"] = []
            anns[idx]["bbox"] = []
            anns[idx]["action"] = []
            # anns[vid][idx]["cross"] = []
            for j in range(flag, len(ped_info_raw)):
                if ped_info_raw[j][1] == pids[i]:
                    ele = ped_info_raw[j]
                    t = int(ele[0].split('.')[0])
                    # box = list([ele[3], ele[2], ele[3] + ele[5], ele[2] + ele[4]])
                    box = list(map(round, [ele[3], ele[2], ele[3] + ele[5], ele[2] + ele[4]]))
                    box = list(map(float, box))
                    action = 1 if ele[6] in ["walking", 'running'] else 0
                    anns[idx]['frames'].append(t)
                    anns[idx]['bbox'].append(box)
                    anns[idx]['action'].append(action)
                else:
                    flag += len(anns[idx]["frames"])
                    break
            anns[idx]['video_number'] = vid
            anns[idx]['old_id'] = vid + f'_{pids[i]}'

    return anns


def add_trans_label_titan(anns, verbose=False) -> None:
    """
    Add labels to show the time (number of frames)
    away from next action transition
    """
    all_wts = 0  # walking to standing
    all_stw = 0  # standing to walking
    pids = list(anns.keys())
    for idx in pids:
        action = anns[idx]['action']
        frames = anns[idx]['frames']
        n_frames = len(frames)
        anns[idx]['next_transition'] = []
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
                anns[idx]['next_transition'].append(next_trans_ped - t)
            else:
                anns[idx]['next_transition'].append(None)
    if verbose:
        print('----------------------------------------------------------------')
        print("TITAN:")
        print(f'Total number of standing to walking transitions (raw): {all_stw}')
        print(f'Total number of walking to standing transitions  (raw): {all_wts}')

    return None


def build_ped_dataset_titan(anns_dir, split_vids_path, image_set="all", verbose=False) -> dict:
    """
    Build pedestrian dataset from TITAN annotations
    """
    assert image_set in ['train', 'test', 'val', "all"], "Image set should be train, test or val"
    vids = get_split_vids_titan(split_vids_path, image_set)
    ped_dataset = convert_anns_titan(anns_dir, vids)
    add_trans_label_titan(ped_dataset, verbose=verbose)

    return ped_dataset


class TitanTransDataset:
    """
     dataset class for transition-related pedestrian samples in TITAN
    """

    def __init__(self, anns_dir, split_vids_path, image_set="all", verbose=False):
        assert image_set in ['train', 'test', 'val', "all"], " Name should be train, test, val or all"
        self.dataset = build_ped_dataset_titan(anns_dir, split_vids_path, image_set, verbose)
        self.name = image_set

    def extract_trans_frame(self, mode="GO", verbose=False) -> dict:
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
            # old_id = copy.deepcopy(dataset[idx]['old_id'])
            # cross = copy.deepcopy(dataset[idx]['cross'])
            next_transition = copy.deepcopy(dataset[idx]["next_transition"])
            for i in range(len(frames)):
                key = None
                if mode == "GO":
                    if next_transition[i] == 0 and action[i] == 1:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "TG_" + new_id
                        old_id = copy.deepcopy(dataset[idx]['old_id'])
                if mode == "STOP":
                    if next_transition[i] == 0 and action[i] == 0:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "TS_" + new_id
                        old_id = copy.deepcopy(dataset[idx]['old_id'])
                if key is not None:
                    samples[key] = {}
                    samples[key]["source"] = "TITAN"
                    samples[key]["old_id"] = old_id
                    samples[key]['video_number'] = vid_id
                    samples[key]['frame'] = frames[i]
                    samples[key]['bbox'] = bbox[i]
                    samples[key]['action'] = action[i]
                    # samples[key]['cross'] = cross[i]
        if verbose:
            print(f"Extract {len(samples.keys())} {mode} sample frames from TITAN {self.name} set")

        return samples

    def extract_trans_history(self, mode="GO", fps=10, verbose=False) -> dict:
        """
        Extract the whole history of pedestrian up to the frame when transition happens
        :params: mode: target transition type, "GO" or "STOP"
                 fps: frame-per-second, sampling rate of extracted sequences, default 30
                 verbose: optional printing of sample statistics
        """
        dataset = self.dataset
        assert mode in ["GO", "STOP"], "Transition type should be STOP or GO"
        ids = list(dataset.keys())
        samples = {}
        j = 0
        step = 10 // fps
        assert isinstance(step, int)
        for idx in ids:
            vid_id = copy.deepcopy(dataset[idx]['video_number'])
            frames = copy.deepcopy(dataset[idx]['frames'])
            bbox = copy.deepcopy(dataset[idx]['bbox'])
            action = copy.deepcopy(dataset[idx]['action'])
            # old_id = copy.deepcopy(dataset[idx]['old_id'])
            # cross = copy.deepcopy(dataset[idx]['cross'])
            next_transition = copy.deepcopy(dataset[idx]["next_transition"])
            for i in range(len(frames)):
                key = None
                if mode == "GO":
                    if next_transition[i] == 0 and action[i] == 1:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "TG_" + new_id
                        old_id = copy.deepcopy(dataset[idx]['old_id'])
                if mode == "STOP":
                    if next_transition[i] == 0 and action[i] == 0:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "TS_" + new_id
                        old_id = copy.deepcopy(dataset[idx]['old_id'])
                if key is not None:
                    samples[key] = {}
                    samples[key]["source"] = "TITAN"
                    samples[key]["old_id"] = old_id
                    samples[key]['video_number'] = vid_id
                    samples[key]['frame'] = frames[i::-step]
                    samples[key]['frame'].reverse()
                    samples[key]['bbox'] = bbox[i::-step]
                    samples[key]['bbox'].reverse()
                    samples[key]['action'] = action[i::-step]
                    samples[key]['action'].reverse()
                    # samples[key]['cross'] = cross[i::-step]
                    # samples[key]['cross'].reverse()
        if verbose:
            keys = list(samples.keys())
            pids = []
            num_frames = 0
            for k in keys:
                pids.append(samples[k]['old_id'])
                num_frames += len(samples[k]['frame'])
            print(f"Extract {len(pids)} {mode} history samples from {self.name} dataset in TITAN,")
            print(f"samples contain {len(set(pids))} unique pedestrians and {num_frames} frames.")

        return samples
