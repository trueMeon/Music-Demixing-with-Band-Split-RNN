import logging
import torch
from torch.utils.data import Dataset
import torchaudio
import random
from collections import defaultdict
from pathlib import Path
import typing as tp
from tqdm import tqdm


log = logging.getLogger(__name__)

class PreloadSourceSeparationDataset(Dataset):
    """
    Dataset class for working with train/validation data from MUSDB18 dataset.
    """
    TARGETS: tp.Set[str] = {'vocals', 'bass', 'drums', 'other'}
    EXTENSIONS: tp.Set[str] = {'.wav', '.mp3'}

    def __init__(
            self,
            file_dir: str,
            txt_dir: str = None,
            target: str = 'vocals',
            is_mono: bool = False,
            is_training: bool = True,
            sr: int = 44100,
            silent_prob: float = 0.1,
            mix_prob: float = 0.1,
            mix_tgt_prob: float = 0.5,
            mix_gain_scale: tp.Tuple[float, float] = [-10, 10],
            mix_chunk_in_secs: int = 3,
    ):
        self.file_dir = Path(file_dir)
        self.is_training = is_training
        self.target = target
        self.sr = sr
        self.txt_dir = txt_dir
        self.is_mono = is_mono
        self.files = []
        self.stems_samples = defaultdict(list)

        # augmentations
        self.silent_prob = silent_prob
        self.mix_prob = mix_prob
        self.mix_tgt_prob = mix_tgt_prob
        self.mix_gain_scale = mix_gain_scale
        self.mix_chunk_size = sr * mix_chunk_in_secs

        self._load_files()

    def _txt_path(self, target: str) -> Path:
        assert self.txt_dir is not None, "'txt_dir' isn't specified"
        mode = 'train' if self.is_training else 'valid'
        return Path(self.txt_dir) / f"{target}_{mode}.txt"

    def _files_offsets(self, target: str):
        offsets = defaultdict(list)

        with open(self._txt_path(target), "r") as file:
            for line in file.readlines():
                file_name, start_idx, end_idx = line.split("\t")
                offsets[file_name].append((int(start_idx), int(end_idx)))
        
        return offsets
    
    def _file_waveform(self, path: str) -> torch.Tensor:
        assert Path(path).is_file(), f"There is no such file - {path}."
        
        y, sr = torchaudio.load(
            path,
            channels_first=True
        )

        assert sr == self.sr, f"Sampling rate should be equal {self.sr}, not {sr}."
        if self.is_mono:
            y = torch.mean(y, dim=0, keepdim=True)

        return y

    def _load_files(self):
        for target in self.TARGETS:
            if not self.is_training and target != self.target:
                continue
            
            mode = "training" if self.is_training else "validation"
            log.info(f"Preloading {target} stem samples for {mode}")
            
            for file_name, offsets in tqdm(self._files_offsets(target).items()):
                filepath_template = str(self.file_dir / "train" / file_name / "{}.wav")

                if target == self.target:
                    mix_waveform = self._file_waveform(filepath_template.format("mixture"))
                    tgt_waveform = self._file_waveform(filepath_template.format(target))

                    for seg_start, seg_end in offsets:
                        mix_segment = mix_waveform[:, seg_start:seg_end]
                        tgt_segment = tgt_waveform[:, seg_start:seg_end]

                        # max_norm = max(
                        #     mix_segment.abs().max(), tgt_segment.abs().max()
                        # )
                        # mix_segment /= max_norm
                        # tgt_segment /= max_norm

                        self.files.append((mix_segment, tgt_segment))
                        self.stems_samples[target].append(tgt_segment)
                else:
                    tgt_waveform = self._file_waveform(filepath_template.format(target))

                    for seg_start, seg_end in offsets:
                        tgt_segment = tgt_waveform[:, seg_start:seg_end]
                        # tgt_segment /= tgt_segment.abs().max()
                        self.stems_samples[target].append(tgt_segment)
                
    @staticmethod
    def _imitate_silent_segments(
            mix_segment: torch.Tensor,
            tgt_segment: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Returns mixture without target and a tensor of zeros the same length as the target (silent segment)
        """
        return (
            mix_segment - tgt_segment,
            torch.zeros_like(tgt_segment)
        )
    
    @staticmethod
    def _db2amp(db):
        return 10 ** (db / 20)
    
    def _random_chunk(self, segment: torch.Tensor) -> torch.Tensor:
        assert segment.shape[1] >= self.mix_chunk_size, "Length of the segment is less than mix_chunk_size"

        start = random.randrange(0, segment.shape[1] - self.mix_chunk_size)
        end = start + self.mix_chunk_size
        return segment[..., start:end]
    
    def _random_corrensponding_chunks(
            self, 
            mix_segment: torch.Tensor, 
            tgt_segment: torch.Tensor,
        )  -> tp.Tuple[torch.Tensor, torch.Tensor]:
        assert mix_segment.shape[1] == tgt_segment.shape[1], "Lengths of the segments aren't equal to each other"
        assert mix_segment.shape[1] >= self.mix_chunk_size, "Length of the segments is less than mix_chunk_size"
        
        start = random.randrange(0, mix_segment.shape[1] - self.mix_chunk_size)
        end = start + self.mix_chunk_size
        return (mix_segment[..., start:end], tgt_segment[..., start:end])

    def _mix_segments(
            self,
            tgt_segment: torch.Tensor,
            mix_tgt_too: bool,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Creating new mixture and new target from target file and random multiple sources
        """
        targets = self.TARGETS

        if not mix_tgt_too:
            targets = targets.difference(self.target)

        n_sources = random.randrange(1, len(targets) + 1)
        # decide which sources to mix
        targets_to_add = random.sample(
            self.TARGETS, n_sources
        )
        # create new mix segment
        tgt_segment *= self._db2amp(random.uniform(*self.mix_gain_scale))
        mix_segment = tgt_segment.clone()
        for target in targets_to_add:
            # get random file to mix source from
            random_segment = random.choice(self.stems_samples[target]) * self._db2amp(random.uniform(*self.mix_gain_scale))
            
            if random_segment is tgt_segment:
                continue

            mix_segment += random_segment
            if target == self.target:
                tgt_segment += random_segment.clone()
        return (
            mix_segment, tgt_segment
        )
    
    def _mixed_sample(self, tgt_segment: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        C, _ = tgt_segment.shape
        new_mix_segment, new_tgt_segment = (
            torch.zeros([C, self.mix_chunk_size]), 
            torch.zeros([C, self.mix_chunk_size])
        )

        for target in self.TARGETS:
            if random.random() < self.silent_prob:
                continue
            
            if target != self.target:
                new_mix_segment += self._random_chunk(
                    random.choice(self.stems_samples[target]) 
                    * self._db2amp(random.uniform(*self.mix_gain_scale))
                )
                continue
            
            random_tgt_chunk = self._random_chunk(
                    tgt_segment * self._db2amp(random.uniform(*self.mix_gain_scale))
                )
            new_mix_segment += random_tgt_chunk
            new_tgt_segment += random_tgt_chunk

            if random.random() >= self.mix_tgt_prob:
                continue

            random_segment = random.choice(self.stems_samples[target])
            if random_segment is tgt_segment:
                continue

            random_tgt_chunk = self._random_chunk(
                    random_segment * self._db2amp(random.uniform(*self.mix_gain_scale))
                )
            new_mix_segment += random_tgt_chunk
            new_tgt_segment += random_tgt_chunk

        return (new_mix_segment, new_tgt_segment)

    def _augment(
            self,
            mix_segment: torch.Tensor,
            tgt_segment: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if self.is_training:
            # mixing with other sources
            if random.random() < self.mix_prob:
                mix_segment, tgt_segment = self._mixed_sample(tgt_segment)
            else:
                mix_segment, tgt_segment = self._random_corrensponding_chunks(mix_segment, tgt_segment)

        return mix_segment, tgt_segment

    def __getitem__(
            self,
            index: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Each Tensor's output shape: [n_channels, frames_in_segment]
        """
        mix_segment, tgt_segment = self.files[index]
        mix_segment, tgt_segment = self._augment(mix_segment, tgt_segment)

        max_norm = max(mix_segment.abs().max(), tgt_segment.abs().max())
        mix_segment /= max_norm
        tgt_segment /= max_norm

        return (mix_segment, tgt_segment)

    def __len__(self):
        return len(self.files)


class SourceSeparationDataset(Dataset):
    """
    Dataset class for working with train/validation data from MUSDB18 dataset.
    """
    TARGETS: tp.Set[str] = {'vocals', 'bass', 'drums', 'other'}
    EXTENSIONS: tp.Set[str] = {'.wav', '.mp3'}

    def __init__(
            self,
            file_dir: str,
            txt_dir: str = None,
            txt_path: str = None,
            target: str = 'vocals',
            preload_dataset: bool = False,
            is_mono: bool = False,
            is_training: bool = True,
            sr: int = 44100,
            silent_prob: float = 0.1,
            mix_prob: float = 0.1,
            mix_tgt_too: bool = False,
    ):
        self.file_dir = Path(file_dir)
        self.is_training = is_training
        self.target = target
        self.sr = sr

        if txt_path is None and txt_dir is not None:
            mode = 'train' if self.is_training else 'valid'
            self.txt_path = Path(txt_dir) / f"{target}_{mode}.txt"
        elif txt_path is not None and txt_dir is None:
            self.txt_path = Path(txt_path)
        else:
            raise ValueError("You need to specify either 'txt_path' or 'txt_dir'.")

        self.preload_dataset = preload_dataset
        self.is_mono = is_mono
        self.filelist = self.get_filelist()

        # augmentations
        self.silent_prob = silent_prob
        self.mix_prob = mix_prob
        self.mix_tgt_too = mix_tgt_too

    def get_filelist(self) -> tp.List[tp.Tuple[str, tp.Tuple[int, int]]]:
        filename2label = {}
        filelist = []
        i = 0
        for line in tqdm(open(self.txt_path, 'r').readlines()):
            file_name, start_idx, end_idx = line.split('\t')
            if file_name not in filename2label:
                filename2label[file_name] = i
                i += 1
            filepath_template = self.file_dir / "train" / f"{file_name}" / "{}.wav"
            if self.preload_dataset:
                mix_segment, tgt_segment = self.load_files(
                    str(filepath_template), (int(start_idx), int(end_idx))
                )
                filelist.append((mix_segment, tgt_segment))
            else:
                filelist.append(
                    (str(filepath_template), (int(start_idx), int(end_idx)))
                )
        return filelist

    def load_file(
            self,
            file_path: str,
            indices: tp.Tuple[int, int]
    ) -> torch.Tensor:
        """Load a single audio file.
        """
        assert Path(file_path).is_file(), f"There is no such file - {file_path}."

        offset = indices[0]
        num_frames = indices[1] - indices[0]
        y, sr = torchaudio.load(
            file_path,
            frame_offset=offset,
            num_frames=num_frames,
            channels_first=True
        )
        assert sr == self.sr, f"Sampling rate should be equal {self.sr}, not {sr}."
        if self.is_mono:
            y = torch.mean(y, dim=0, keepdim=True)
        return y

    def load_files(
            self, fp_template: str, indices: tp.Tuple[int, int],
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Load audio files for target and mixture and then normalize.
        """
        mix_segment = self.load_file(
            fp_template.format('mixture'), indices
        )
        tgt_segment = self.load_file(
            fp_template.format(self.target), indices
        )

        # Normalize mixture and target
        max_norm = max(
            mix_segment.abs().max(), tgt_segment.abs().max()
        )
        mix_segment /= max_norm
        tgt_segment /= max_norm
        return (
            mix_segment, tgt_segment
        )

    @staticmethod
    def imitate_silent_segments(
            mix_segment: torch.Tensor,
            tgt_segment: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Returns mixture without target and a tensor of zeros the same length as the target (silent segment)
        """
        return (
            mix_segment - tgt_segment,
            torch.zeros_like(tgt_segment)
        )

    def mix_segments(
            self,
            tgt_segment: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Creating new mixture and new target from target file and random multiple sources
        """
        # decide how many sources to mix
        if not self.mix_tgt_too:
            self.TARGETS.discard(self.target)
        n_sources = random.randrange(1, len(self.TARGETS) + 1)
        # decide which sources to mix
        targets_to_add = random.sample(
            self.TARGETS, n_sources
        )
        # create new mix segment
        mix_segment = tgt_segment.clone()
        for target in targets_to_add:
            # get random file to mix source from
            fp_template_to_add, indices_to_add = random.choice(self.filelist)
            segment_to_add = self.load_file(
                fp_template_to_add.format(target), indices_to_add
            )
            mix_segment += segment_to_add
            if target == self.target:
                tgt_segment += segment_to_add
        return (
            mix_segment, tgt_segment
        )

    def augment(
            self,
            mix_segment: torch.Tensor,
            tgt_segment: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if self.is_training:
            # dropping target
            if random.random() < self.silent_prob:
                mix_segment, tgt_segment = self.imitate_silent_segments(
                    mix_segment, tgt_segment
                )
            # mixing with other sources
            if random.random() < self.mix_prob:
                mix_segment, tgt_segment = self.mix_segments(
                    tgt_segment
                )
        return mix_segment, tgt_segment

    def __getitem__(
            self,
            index: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Each Tensor's output shape: [n_channels, frames_in_segment]
        """
        # load files
        if self.preload_dataset:
            mix_segment, tgt_segment = self.filelist[index]
        else:
            mix_segment, tgt_segment = self.load_files(*self.filelist[index])
            # augmentations related to mixing/dropping sources
            mix_segment, tgt_segment = self.augment(mix_segment, tgt_segment)

        return (
            mix_segment, tgt_segment
        )

    def __len__(self):
        return len(self.filelist)


class EvalSourceSeparationDataset(Dataset):
    """
    Dataset class for working with test data from MUSDB18 dataset.
    """
    EXTENSIONS: tp.Set[str] = {'.wav', '.mp3'}

    def __init__(
            self,
            mode: str,
            in_fp: str,
            out_fp: tp.Optional[str] = None,
            target: str = 'vocals',
            is_mono: bool = False,
            sr: int = 44100,
            win_size: float = 3,
            hop_size: float = 0.5,
            *args, **kwargs
    ):
        self.mode = mode

        # files params
        self.in_fp = Path(in_fp)
        self.out_fp = Path(out_fp) if out_fp is not None else None
        self.target = target

        # audio params
        self.is_mono = is_mono
        self.sr = sr
        self.win_size = int(win_size * sr)
        self.hop_size = int(hop_size * sr)
        self.pad_size = self.win_size - self.hop_size

        self.filelist = self.get_filelist()

    def get_test_filelist(self) -> tp.List[tp.Tuple[str, str]]:
        filelist = []
        test_dir = self.in_fp / self.mode

        for fp in test_dir.glob('*'):
            fp_template = str(fp / "{}.wav")
            fp_mix = fp_template.format('mixture')
            fp_tgt = fp_template.format(self.target)
            filelist.append((fp_mix, fp_tgt))

        return filelist

    def get_inference_filelist(self) -> tp.List[tp.Tuple[str, str]]:
        filelist = []
        if self.in_fp.is_file() and self.in_fp.suffix in self.EXTENSIONS:
            self.out_fp = self.out_fp / f"{self.in_fp.stem}_{self.target}.wav"
            filelist.append((self.in_fp, self.out_fp))
        elif self.in_fp.is_dir():
            for in_fp in self.in_fp.glob("*"):
                if in_fp.suffix in self.EXTENSIONS:
                    out_fp = self.out_fp / f"{in_fp.stem}_{self.target}.wav"
                    filelist.append((str(in_fp), str(out_fp)))
        else:
            raise ValueError(f"Can not open the path {self.in_fp}")
        return filelist

    def get_filelist(self) -> tp.List[tp.Tuple[str, str]]:
        if self.mode == 'test':
            filelist = self.get_test_filelist()
        elif self.mode == 'inference':
            filelist = self.get_inference_filelist()
        else:
            raise ValueError(f"Selected mode = '{self.mode}' is invalid")
        return filelist

    def load_file(self, file_path: str) -> torch.Tensor:
        assert Path(file_path).is_file(), f"There is no such file - {file_path}."
        y, sr = torchaudio.load(
            file_path,
            channels_first=True
        )
        if sr != self.sr:
            y = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sr
            )(y)
        # setting to mono if necessary
        if self.is_mono:
            y = torch.mean(y, dim=0, keepdim=True)
        elif y.shape[0] == 1:
            y = y.repeat(2, 1)
        return y

    def __getitem__(
            self, index: int
    ) -> tp.Tuple[torch.Tensor, tp.Union[torch.Tensor, str]]:

        fp_mix, fp_tgt = self.filelist[index]

        y_mix = self.load_file(fp_mix)

        if self.mode == 'test':
            return y_mix, self.load_file(fp_tgt)
        else:
            return y_mix, fp_tgt

    def __len__(self):
        return len(self.filelist)
