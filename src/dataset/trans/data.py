import random
from .jaad_trans import *
from .pie_trans import *
from .titan_trans import *


class TransDataset:
    """
    Unified class for using data from JAAD, PIE and TITAN dataset.
    """
    def __init__(self, data_paths, image_set="all", subset='default', verbose=False):
        dataset = {}
        assert image_set in ['train', 'test', 'val', "all"], " Name should be train, test, val or all"
        for d in list(data_paths.keys()):
            assert d in ['JAAD', 'PIE', 'TITAN'], " Available datasets are JAAD, PIE and TITAN"
            if d == "JAAD":
                dataset['JAAD'] = JaadTransDataset(
                    jaad_anns_path=data_paths['JAAD']['anns'],
                    split_vids_path=data_paths['JAAD']['split'],
                    image_set=image_set,
                    subset=subset, verbose=verbose)
            elif d == "PIE":
                dataset['PIE'] = PieTransDataset(
                    pie_anns_path=data_paths['PIE']['anns'],
                    image_set=image_set, verbose=verbose)
            elif d == "TITAN":
                dataset['TITAN'] = TitanTransDataset(
                    anns_dir=data_paths['TITAN']['anns'],
                    split_vids_path=data_paths['TITAN']['split'],
                    image_set=image_set, verbose=verbose)

        self.dataset = dataset
        self.name = image_set
        self.subset = subset

    def __repr__(self):
        return f"TransDataset(image_set={self.name}, jaad_subset={self.subset})"

    def extract_trans_frame(self, mode="GO", frame_ahead=0, fps=10, verbose=False) -> dict:
        ds = list(self.dataset.keys())
        samples = {}
        for d in ds:
            samples_new = self.dataset[d].extract_trans_frame(mode=mode, frame_ahead=frame_ahead, fps=fps)
            samples.update(samples_new)
        if verbose:
            ids = list(samples.keys())
            pids = []
            for idx in ids:
                pids.append(samples[idx]['old_id'])
            print(f"Extract {len(pids)} {mode} frame samples from {self.name} dataset,")
            print(f"samples contain {len(set(pids))} unique pedestrians.")

        return samples

    def extract_trans_history(self, mode="GO", fps=10, max_frames=None, post_frames=0, verbose=False) -> dict:
        """
        Extract the whole history of pedestrian up to the frame when transition happens
        :params: mode: target transition type, "GO" or "STOP"
                fps: frame-per-second, sampling rate of extracted sequences, default 10
                max_frames: maximum number of frames in one history
                post_frames: number of frames included after the transition
                verbose: optional printing of sample statistics
        """
        assert isinstance(fps, int) and 30 % fps == 0, "impossible fps"
        ds = list(self.dataset.keys())
        samples = {}
        for d in ds:
            samples_new = self.dataset[d].extract_trans_history(mode=mode, fps=fps, max_frames=max_frames,
                                                                post_frames=post_frames)
            samples.update(samples_new)
        if verbose:
            ids = list(samples.keys())
            pids = []
            num_frames = 0
            for idx in ids:
                pids.append(samples[idx]['old_id'])
                num_frames += len(samples[idx]['frame'])
            print(f"Extract {len(pids)} {mode} history samples from {self.name} dataset,")
            print(f"samples contain {len(set(pids))} unique pedestrians and {num_frames} frames.")

        return samples

    def extract_non_trans(self, fps=10, max_frames=None, verbose=False) -> dict:
        assert isinstance(fps, int) and 30 % fps == 0, "impossible fps"
        ds = list(self.dataset.keys())
        samples = {'walking': {}, 'standing': {}}
        for d in ds:
            # Set the number of samples needed in TITAN
            if d == 'TITAN':
                if self.name == 'all':
                    n_titan = 600
                elif self.name == 'train':
                    n_titan = 300
                elif self.name == 'val':
                    n_titan = 200
                else:
                    n_titan = 100
                samples_new = self.dataset[d].extract_non_trans(fps=fps, max_frames=max_frames, max_samples=n_titan)
            else:
                samples_new = self.dataset[d].extract_non_trans(fps=fps, max_frames=max_frames)
            samples['walking'].update(samples_new['walking'])
            samples['standing'].update(samples_new['standing'])
        if verbose:
            keys_w = list(samples['walking'].keys())
            keys_s = list(samples['standing'].keys())
            pid_w = []
            pid_s = []
            n_w = 0
            n_s = 0
            for kw in keys_w:
                pid_w.append(samples['walking'][kw]['old_id'])
                n_w += len(samples['walking'][kw]['frame'])
            for ks in keys_s:
                pid_s.append(samples['standing'][ks]['old_id'])
                n_s += len(samples['standing'][ks]['frame'])
            print(f"Extract Non-transition samples from {self.name} dataset  :")
            print(f"Walking: {len(pid_w)} samples,  {len(set(pid_w))} unique pedestrians and {n_w} frames.")
            print(f"Standing: {len(pid_s)} samples,  {len(set(pid_s))} unique pedestrians and {n_s} frames.")

        return samples


def balance_frame_sample(samples, seed=99, balancing_ratio=1, verbose=True) -> dict:
    """
    Balance positive and negative frames samples by given ratio.
    """
    random.seed(seed)
    ids = list(samples.keys())
    ids_j = []
    ids_p = []
    ids_t = []
    ids_new = []
    for k in ids:
        if samples[k]['source'] == "JAAD":
            ids_j.append(k)
        elif samples[k]['source'] == "PIE":
            ids_p.append(k)
        elif samples[k]['source'] == "TITAN":
            ids_t.append(k)
    # balance positive and negative samples within each dataset
    for _ids in [ids_j, ids_p, ids_t]:
        if len(_ids) == 0:
            continue
        source = samples[_ids[0]]['source']
        ps = []
        ns = []
        for i in range(len(_ids)):
            key = _ids[i]
            if samples[key]['trans_label'] == 1:
                ps.append(key)
            else:
                ns.append(key)
        size = int(min(len(ps), len(ns)) * balancing_ratio)
        size = max(len(ps), len(ns)) if size > max(len(ps), len(ns)) else size
        ps_new = random.sample(ps, size) if len(ps) > len(ns) else ps
        ns_new = random.sample(ns, size) if len(ps) < len(ns) else ns
        if verbose:
            print(f"Perform sample balancing for {source}:")
            print(f'Orignal samples: P {len(ps)} , N {len(ns)}')
            print(f'Balanced samples:P {len(ps_new)}, N {len(ns_new)}')
        ids_new = ids_new + ps_new + ns_new
    # balanced samples
    random.shuffle(ids_new)
    samples_new = {}
    for key in ids_new:
        samples_new[key] = copy.deepcopy(samples[key])

    return samples_new


def extract_pred_frame(trans, non_trans=None, pred_ahead=0, balancing_ratio=None,
                       bbox_min=0, seed=None, verbose=False) -> dict:
    """
    Extract the frames in history for transition prediction task.
    :params: trans: transition history samples, i.e. GO or STOP
             non-trans: history samples containing no transitions
             pred_ahead: frame to predicted in advance, whether the trnasition occur in ~ frames.
             balancing_ratio: ratio between positive and negative frame instances
             bbox_min: minimum width of the pedestrian bounding box
             seed: random used during balancing
             verbose: optional printing
    """
    assert isinstance(pred_ahead, int) and pred_ahead >= 0, "Invalid prediction length."
    ids_trans = list(trans.keys())
    samples = {}
    n_1 = 0
    if isinstance(bbox_min, int):
        bbox_min = (bbox_min, bbox_min)
    for idx in ids_trans:
        frames = copy.deepcopy(trans[idx]['frame'])
        bbox = copy.deepcopy(trans[idx]['bbox'])
        action = copy.deepcopy(trans[idx]['action'])
        d_pre = trans[idx]['pre_state']
        n_frames = len(frames)
        fps = trans[idx]['fps']
        source = trans[idx]['source']
        step = 60 // fps if source == 'TITAN' else 30 // fps
        for i in range(max(0, n_frames - d_pre), n_frames - 1):
            if abs(bbox[i][2] - bbox[i][0]) < bbox_min[0]:
                continue
            key = idx + f"_f{frames[i]}"
            TTE = (frames[-1] - frames[i]) / (step * fps)
            if TTE > pred_ahead / fps:
                trans_label = 0
                key = None
            else:
                trans_label = 1
                n_1 += 1
            if key is not None:
                samples[key] = {}
                samples[key]['source'] = trans[idx]['source']
                if samples[key]['source'] == 'PIE':
                    samples[key]['set_number'] = trans[idx]['set_number']
                samples[key]['video_number'] = trans[idx]['video_number']
                samples[key]['frame'] = frames[i]
                samples[key]['bbox'] = bbox[i]
                samples[key]['action'] = action[i]
                samples[key]['trans_label'] = trans_label
                samples[key]['TTE'] = TTE
    # negative instances from all examples
    if non_trans is not None:
        action_type = 'walking' if trans[ids_trans[0]]['type'] == 'STOP' else 'standing'
        ids_non_trans = list(non_trans[action_type].keys())
        for idx in ids_non_trans:
            frames = copy.deepcopy(non_trans[action_type][idx]['frame'])
            bbox = copy.deepcopy(non_trans[action_type][idx]['bbox'])
            action = copy.deepcopy(non_trans[action_type][idx]['action'])
            for i in range(len(frames)):
                if abs(bbox[i][2] - bbox[i][0]) < bbox_min[1]:
                    continue
                key = idx + f"_f{frames[i]}"
                samples[key] = {}
                samples[key]['source'] = non_trans[action_type][idx]['source']
                if samples[key]['source'] == 'PIE':
                    samples[key]['set_number'] = non_trans[action_type][idx]['set_number']
                samples[key]['video_number'] = non_trans[action_type][idx]['video_number']
                samples[key]['frame'] = frames[i]
                samples[key]['bbox'] = bbox[i]
                samples[key]['action'] = action[i]
                samples[key]['trans_label'] = 0
                samples[key]['TTE'] = -1

    if verbose:
        if n_1 > 0:
            ratio = (len(samples.keys()) - n_1) / n_1
        else:
            ratio = 999.99
        print(f'Extract {len(samples.keys())} frame samples from {len(trans.keys())} history sequences.')
        print('1/0 ratio:  1 : {:.2f}'.format(ratio))
        print(f'predicting-ahead frames: {pred_ahead}')

    if balancing_ratio is not None:
        samples = balance_frame_sample(samples=samples, seed=seed, balancing_ratio=balancing_ratio, verbose=verbose)

    return samples


def record_trans_stats(samples):
    j_pre = []
    p_pre = []
    t_pre = []
    j_pos = []
    p_pos = []
    t_pos = []
    fps = []
    ids = list(samples.keys())
    for idx in ids:
        fps.append(samples[idx]['fps'])
        if samples[idx]['source'] == 'JAAD':
            j_pre.append(samples[idx]['pre_state'])
            j_pos.append(samples[idx]['post_state'])
        elif samples[idx]['source'] == 'PIE':
            p_pre.append(samples[idx]['pre_state'])
            p_pos.append(samples[idx]['post_state'])
        elif samples[idx]['source'] == 'TITAN':
            t_pre.append(samples[idx]['pre_state'])
            t_pos.append(samples[idx]['post_state'])
    pre = {'JAAD': j_pre, 'PIE': p_pre, 'TITAN': t_pre}
    pos = {'JAAD': j_pos, 'PIE': p_pos, 'TITAN': t_pos}
    trans_stats = {'Pre': pre, 'Pos': pos, 'fps': fps}

    return trans_stats


def sequence_to_frame(sequence):
    samples = {}
    frames = copy.deepcopy(sequence['frame'])
    bbox = copy.deepcopy(sequence['bbox'])
    action = copy.deepcopy(sequence['action'])
    n_frames = len(frames)
    for i in range(n_frames):
        key = f'f_{frames[i]}'
        samples[key] = {}
        samples[key]['source'] = sequence['source']
        if samples[key]['source'] == 'PIE':
            samples[key]['set_number'] = sequence['set_number']
        samples[key]['video_number'] = sequence['video_number']
        samples[key]['frame'] = frames[i]
        samples[key]['bbox'] = bbox[i]
        samples[key]['action'] = action[i]

    return samples


def extract_pred_sequence(trans, non_trans=None, pred_ahead=0, balancing_ratio=None,
                          bbox_min=0, max_frames=None, seed=None, verbose=False) -> dict:
    """
    Extract  sequences for transition prediction task.
    :params: trans: transition history samples, i.e. GO or STOP
             non-trans: history samples containing no transitions
             pred_ahead: frame to predicted in advance, whether the trnasition occur in ~ frames.
             balancing_ratio: ratio between positive and negative frame instances
             bbox_min: minimum width of the pedestrian bounding box
             max_frames: maximum frames in one sequence sample
             seed: random used during balancing
             verbose: optional printing
    """
    assert isinstance(pred_ahead, int) and pred_ahead >= 0, "Invalid prediction length."
    ids_trans = list(trans.keys())
    samples = {}
    n_1 = 0
    if isinstance(bbox_min, int):
        bbox_min = (bbox_min, bbox_min)
    for idx in ids_trans:
        frames = copy.deepcopy(trans[idx]['frame'])
        bbox = copy.deepcopy(trans[idx]['bbox'])
        action = copy.deepcopy(trans[idx]['action'])
        d_pre = trans[idx]['pre_state']
        n_frames = len(frames)
        fps = trans[idx]['fps']
        source = trans[idx]['source']
        step = 60 // fps if source == 'TITAN' else 30 // fps
        for i in range(max(0, n_frames - d_pre), n_frames - 1):
            if abs(bbox[i][2] - bbox[i][0]) < bbox_min[0]:
                continue
            key = idx + f"_f{frames[i]}"
            TTE = (frames[-1] - frames[i]) / (step * fps)
            if TTE > pred_ahead / fps:
                trans_label = 0
                key = None
            else:
                trans_label = 1
                n_1 += 1
            if key is not None:
                samples[key] = {}
                samples[key]['source'] = trans[idx]['source']
                if samples[key]['source'] == 'PIE':
                    samples[key]['set_number'] = trans[idx]['set_number']
                samples[key]['video_number'] = trans[idx]['video_number']
                # t = 0 if max_frames is None else i - max_frames + 1
                if max_frames is None:
                    t = 0
                else:
                    if i < max_frames - 1:
                        t = 0
                    else:
                        t = i - max_frames + 1
                samples[key]['frame'] = frames[t:i + 1]
                samples[key]['bbox'] = bbox[t:i + 1]
                samples[key]['action'] = action[t:i + 1]
                samples[key]['trans_label'] = trans_label
                samples[key]['TTE'] = TTE
    # negative instances from all examples
    if non_trans is not None:
        action_type = 'walking' if trans[ids_trans[0]]['type'] == 'STOP' else 'standing'
        ids_non_trans = list(non_trans[action_type].keys())
        for idx in ids_non_trans:
            frames = copy.deepcopy(non_trans[action_type][idx]['frame'])
            bbox = copy.deepcopy(non_trans[action_type][idx]['bbox'])
            action = copy.deepcopy(non_trans[action_type][idx]['action'])
            for i in range(len(frames)):
                if abs(bbox[i][2] - bbox[i][0]) < bbox_min[1]:
                    continue
                key = idx + f"_f{frames[i]}"
                samples[key] = {}
                samples[key]['source'] = non_trans[action_type][idx]['source']
                if samples[key]['source'] == 'PIE':
                    samples[key]['set_number'] = non_trans[action_type][idx]['set_number']
                samples[key]['video_number'] = non_trans[action_type][idx]['video_number']
                # t = 0 if max_frames is None else i - max_frames + 1
                if max_frames is None:
                    t = 0
                else:
                    if i < max_frames - 1:
                        t = 0
                    else:
                        t = i - max_frames + 1
                samples[key]['frame'] = frames[t:i + 1]
                samples[key]['bbox'] = bbox[t:i + 1]
                samples[key]['action'] = action[t:i + 1]
                samples[key]['trans_label'] = 0
                samples[key]['TTE'] = -1

    if verbose:
        if n_1 > 0:
            ratio = (len(samples.keys()) - n_1) / n_1
        else:
            ratio = 999.99
        print(f'Extract {len(samples.keys())} sequence samples from {len(trans.keys())} history.')
        print('1/0 ratio:  1 : {:.2f}'.format(ratio))
        print(f'predicting-ahead frames: {pred_ahead}')

    if balancing_ratio is not None:
        samples = balance_frame_sample(samples=samples, seed=seed, balancing_ratio=balancing_ratio, verbose=verbose)

    return samples
