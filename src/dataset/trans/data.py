import random
from .jaad_trans import *
from .pie_trans import *
from .titan_trans import *


class TransDataset:

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

    def extract_trans_history(self, mode="GO", fps=10, max_frames=None, verbose=False) -> dict:
        """
        Extract the whole history of pedestrian up to the frame when transition happens
        :params: mode: target transition type, "GO" or "STOP"
                fps: frame-per-second, sampling rate of extracted sequences, default 10
                verbose: optional printing of sample statistics
        """
        assert isinstance(fps, int) and 30 % fps == 0, "impossible fps"
        ds = list(self.dataset.keys())
        samples = {}
        for d in ds:
            samples_new = self.dataset[d].extract_trans_history(mode=mode, fps=fps, max_frames=max_frames)
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


def balance_frame_sample(samples, seed, balancing_ratio=1, verbose=True):
    random.seed(seed)
    ids = list(samples.keys())
    ps = []
    ns = []
    for i in range(len(ids)):
        key = ids[i]
        if samples[key]['trans_label'] == 1:
            ps.append(key)
        else:
            ns.append(key)
    size = int(min(len(ps), len(ns)) * balancing_ratio)
    size = max(len(ps), len(ns)) if size > max(len(ps), len(ns)) else size
    ps_new = random.sample(ps, size) if len(ps) > len(ns) else ps
    ns_new = random.sample(ns, size) if len(ps) < len(ns) else ns
    ids_new = ps_new + ns_new
    random.shuffle(ids_new)
    samples_new = {}
    for key in ids_new:
        samples_new[key] = copy.deepcopy(samples[key])

    if verbose:
        print("Perform sample balancing:")
        print(f'Orignal samples: P {len(ps)} , N {len(ns)}')
        print(f'Balanced samples:P {len(ps_new)}, N {len(ns_new)}')

    return samples_new


def extract_pred_frame(history, pred_ahead=0, balancing_ratio=None, seed=None, verbose=False) -> dict:
    assert isinstance(pred_ahead, int) and pred_ahead >= 0, "Invalid prediction length."
    ids = list(history.keys())
    samples = {}
    n_1 = 0
    for idx in ids:
        frames = copy.deepcopy(history[idx]['frame'])
        bbox = copy.deepcopy(history[idx]['bbox'])
        action = copy.deepcopy(history[idx]['action'])
        d_pre = history[idx]['pre_state']
        n_frames = len(frames)
        for i in range(max(0, n_frames - d_pre), n_frames):
            key = idx + f"_f{frames[i]}"
            if pred_ahead < n_frames and frames[i] < frames[n_frames - pred_ahead]:
                trans_label = 0
            else:
                trans_label = 1
                n_1 += 1
            if key is not None:
                samples[key] = {}
                samples[key]['source'] = history[idx]['source']
                if samples[key]['source'] == 'PIE':
                    samples[key]['set_number'] = history[idx]['set_number']
                samples[key]['video_number'] = history[idx]['video_number']
                samples[key]['frame'] = frames[i]
                samples[key]['bbox'] = bbox[i]
                samples[key]['action'] = action[i]
                samples[key]['trans_label'] = trans_label
    if verbose:
        print(f'Extract {len(samples.keys())} frame samples from {len(history.keys())} history sequences.')
        print('1/0 ratio:  1 : {:.2f}'.format((len(samples.keys()) - n_1) / n_1))
        print(f'predicting-ahead frames: {pred_ahead}')
        
    if balancing_ratio is not None:
        samples = balance_frame_sample(samples=samples, seed=seed, balancing_ratio=balancing_ratio, verbose=verbose)

    return samples
